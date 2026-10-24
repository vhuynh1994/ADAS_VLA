"""PC simulation of adas-vlm-v3 as it runs on the board (deploy/qwen_vl.py): the decide() interface of the project's
VLMs, so training.evaluate scores it like any other VLM.

What the board computes, step by step: the decision prompt rendered with the chat template and split around the
image, the image patchified at 560x308, ViT, LUT embedding, the four LLM parts run like Genie (AR128 prefill, AR1
decode, KV cache of 1024), plain greedy decoding stopped at '"reason"' (the server's `fast` config). In float this
is the reference the board is compared with; `wbits` / `kv_bits` simulate the precisions of the quantized graphs
(symmetric 4/8-bit weights per output channel or LPBQ blocks, 8-bit per-tensor KV cache) before building them.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..reasoning.parser import parse_decision
from ..reasoning.prompts import DECISION_PREFIX, decision_messages
from ..reasoning.vlm import LONG_ORDER, REASON_KEY, close_before_reason
from . import qwen_vl as qv

IMAGE_SPLIT = "<|vision_start|><|image_pad|><|vision_end|>"


def split_prompt(text: str) -> tuple[str, str]:
    """Chat-template text with one image -> (text up to <|vision_start|>, text from <|vision_end|>)."""
    i = text.index(IMAGE_SPLIT)
    end = i + len(IMAGE_SPLIT) - len("<|vision_end|>")
    return text[:i] + "<|vision_start|>", text[end:]


def fake_quant(w: torch.Tensor, bits: int, block: int | None = None) -> torch.Tensor:
    """Symmetric signed weights [out, in] quantized and dequantized per output channel, or LPBQ (QNN
    BLOCKWISE_EXPANSION): blocks of `block` input channels, each block scale = channel scale x an integer 1..16."""
    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    if block is None:
        s = wf.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        return ((wf / s).round().clamp(-qmax - 1, qmax) * s).to(w.dtype)
    o, i = wf.shape
    wb = wf.reshape(o, i // block, block)
    sb = wb.abs().amax(2, keepdim=True).clamp_min(1e-12) / qmax
    sc = sb.amax(1, keepdim=True) / 16
    sb = (sb / sc).ceil().clamp(1, 16) * sc
    return ((wb / sb).round().clamp(-qmax - 1, qmax) * sb).reshape(o, i).to(w.dtype)


def llm_convs(part: qv.DecoderPart):
    for layer in part.layers:
        yield from (*layer.q, *layer.k, *layer.v, layer.o, layer.gate, layer.up, layer.down)


@torch.no_grad()
def quantize_weights(parts: list[qv.DecoderPart], bits: int | None, block: int | None = None,
                     head_bits: int = 8) -> None:
    for part in parts:
        for conv in llm_convs(part) if bits else ():
            conv.weight[:, :, 0, 0] = fake_quant(conv.weight[:, :, 0, 0], bits,
                                                 block if conv.in_channels % (block or 1) == 0 else None)
        if part.last:
            w = part.lm_head.weight
            for a in range(0, w.shape[0], 16384):
                w[a:a + 16384, :, 0, 0] = fake_quant(w[a:a + 16384, :, 0, 0], head_bits)


class KVQuant:
    """8-bit symmetric per-tensor K / V cache of each layer, scales from the absolute maxima seen while calibrating."""

    def __init__(self, bits: int = 8):
        self.qmax = 2 ** (bits - 1) - 1
        self.absmax: dict = defaultdict(float)
        self.calibrating = True

    def __call__(self, layer: int, kind: str, t: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self.absmax[(layer, kind)] = max(self.absmax[(layer, kind)], float(t.abs().max()))
            return t
        s = self.absmax[(layer, kind)] / self.qmax
        return ((t.float() / s).round().clamp(-self.qmax - 1, self.qmax) * s).to(t.dtype)


class DeployVLM:
    def __init__(self, model_dir: str | Path, device: str = "cuda", dtype=torch.bfloat16, rotate: bool = False,
                 wbits: int | None = None, wblock: int | None = None, kv_bits: int | None = None,
                 max_new_tokens: int = 64, cl: int = 1024, ar: int = 128, rotate_heads: bool = False,
                 gptq: str | Path | None = None, feature_cache: dict | None = None):
        from tokenizers import Tokenizer
        from transformers import AutoProcessor

        self.spec = spec = qv.Spec.from_model(model_dir)
        ck = qv.Checkpoint(model_dir)
        self.rotation = qv.hadamard(spec.dim) if rotate else None
        self.head_rotation = qv.hadamard(spec.head_dim, seed=1) if rotate_heads else None
        self.device, self.dtype, self.max_new_tokens = torch.device(device), dtype, max_new_tokens
        self.feature_cache = feature_cache
        self.vit = None if feature_cache is not None else qv.VisionEncoder(ck, spec, self.rotation, dtype).to(
            self.device).eval()
        ranges = qv.part_ranges(spec)
        qw = None
        if gptq:
            from .gptq import QuantizedWeights

            qw = QuantizedWeights(gptq)
        self.parts = [qv.DecoderPart(ck, spec, r, i == len(ranges) - 1, self.rotation, dtype,
                                     self.head_rotation, qw).to(self.device).eval() for i, r in enumerate(ranges)]
        if wbits or gptq:  # GPTQ weights are already on their grid: only the LM head (8-bit) is rounded here
            quantize_weights(self.parts, None if gptq else wbits, wblock)
        self.kv = KVQuant(kv_bits) if kv_bits else None
        self.gen = qv.Generator(self.parts, spec, cl, ar, self.device, dtype, kv_quant=self.kv)
        self.lut = qv.embedding_table(ck, self.rotation)
        self.vit_in = [t.to(self.device, dtype) for k, t in qv.vit_inputs(spec).items()]
        self.processor = AutoProcessor.from_pretrained(model_dir)  # chat template only
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.first_token = {a: self.tok.encode(a, add_special_tokens=False).ids[0] for a in LONG_ORDER}
        self.last_timing: dict[str, float] = {}

    def prompt(self, image, context_text: str, ego_speed_kmh: float, cruise_speed_kmh: float) -> tuple[list, list]:
        text = self.processor.apply_chat_template(
            decision_messages(image, context_text, ego_speed_kmh, cruise_speed_kmh), tokenize=False,
            add_generation_prompt=True)
        before, after = split_prompt(text)
        enc = lambda s: self.tok.encode(s, add_special_tokens=False).ids  # noqa: E731
        return enc(before), enc(after + DECISION_PREFIX)

    @staticmethod
    def image_key(rgb: np.ndarray) -> str:
        return hashlib.sha1(rgb.tobytes()).hexdigest()

    @torch.no_grad()
    def image_features(self, image) -> torch.Tensor:
        rgb = qv.resize_rgb(image, self.spec)
        if self.feature_cache is not None:  # e.g. float32 ViT features computed beforehand (vit-cache)
            return self.feature_cache[self.image_key(rgb)].to(self.device, self.dtype)
        pix = torch.from_numpy(qv.patchify(rgb, self.spec)).to(self.device, self.dtype)
        return self.vit(pix, *self.vit_in)

    @torch.no_grad()
    def decide(self, image, context_text: str, ego_speed_kmh: float, cruise_speed_kmh: float,
               max_speed_kmh: float = 130.0, prev_image=None):
        t0 = time.perf_counter()
        ids_b, ids_a = self.prompt(image, context_text, ego_speed_kmh, cruise_speed_kmh)
        feats = self.image_features(image)
        t_vit = time.perf_counter()
        lut = lambda ids: self.lut[ids].to(self.device, self.dtype)  # noqa: E731
        embeds = torch.cat([lut(ids_b), feats, lut(ids_a)])
        self.gen.reset()
        logits = self.gen.prefill(embeds, qv.prompt_positions(len(ids_b), len(ids_a), self.spec))
        t_first = time.perf_counter()
        p = torch.softmax(logits, -1)
        raw_p = {a: float(p[t]) for a, t in self.first_token.items()}
        total = sum(raw_p.values()) or 1.0
        tokens = [int(logits.argmax())]
        text = self.tok.decode(tokens)
        while len(tokens) < self.max_new_tokens and tokens[-1] not in self.spec.eos and REASON_KEY not in text:
            tokens.append(int(self.gen.decode(self.lut[tokens[-1]].to(self.device, self.dtype)).argmax()))
            text = self.tok.decode([t for t in tokens if t not in self.spec.eos])
        latency = time.perf_counter() - t0
        self.last_timing = {"vit_s": t_vit - t0, "first_token_s": t_first - t0, "total_s": latency,
                            "tokens": len(tokens), "prompt_tokens": len(embeds)}
        raw = close_before_reason(DECISION_PREFIX + text)
        decision = parse_decision(raw, ego_speed_kmh, max_speed_kmh)
        if decision is not None:
            decision.latency_s = latency
            decision.action_probs = {a: v / total for a, v in raw_p.items()}
        return decision, raw, latency
