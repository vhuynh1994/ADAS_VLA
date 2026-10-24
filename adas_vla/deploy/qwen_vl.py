"""Qwen2.5-VL (the fine-tuned models/adas-vlm-v3) in the form it runs on the Qualcomm HTP under Genie.

Plain PyTorch modules whose forward() is the graph exported to ONNX for QAIRT, with the tensor names and layouts of
the Qualcomm AI Hub Genie bundle of Qwen2.5-VL-7B (so the board's Genie runs it with the same configs):

  RGB 560x308 --patchify (host)--> pixel_values [880, 1176]
  VisionEncoder (HTP): pixel_values, position_ids_cos/sin [880, 40], window/full_attention_mask [1, 880, 880]
                       -> image_features [220, 2048]
  embedding LUT (host): token ids -> [n, 2048]; the 220 <|image_pad|> rows are the image features
  DecoderPart i of 4 (HTP), graphs ar128_cl1024_i_of_4 / ar1_cl1024_i_of_4:
      inputs_embeds | hidden_<a> [1, N, 2048], attention_mask [1, 1, N, CL], position_ids_cos/sin [1, 1, N, 64],
      past_key_<l>_in [2, 1, 128, CL-N], past_value_<l>_in [2, 1, CL-N, 128] for each layer l of the part
      -> hidden_<b> (logits [1, N, 151936] in the last part), past_key_<l>_out [2, 1, 128, N],
         past_value_<l>_out [2, 1, N, 128] (only the N new entries: the runtime keeps the cache)

HTP adaptations as in the AI Hub recipe: Linear -> 1x1 Conv2d on [1, C, 1, N], attention split per head (SHA) in the
LLM, RoPE from precomputed cos/sin of half the head dim, additive masks as inputs, Q scaled before Q.K^T, K cache
transposed. Weights are read tensor by tensor from the safetensors file, so a part is built without loading the
rest of the model (the whole model in float32 is 15 GB; the laptop has 31 GB of RAM).

`rotation` (for 4-bit weights): fold the RMSNorm weights into the next linear layers and rotate the residual stream
with a randomized Hadamard matrix R (QuaRot / SpinQuant "R1"): embedding table E @ R, layers reading the residual
W diag(g) R, layers writing it R^T W, LM head (E diag(g_final)) R, the ViT's last linear writes features @ R. The float
model computes the same function; outlier channels are spread over all 2048, which 4-bit per-channel weights and
16-bit per-tensor activations need.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
VIT_MASK = -1000.0  # additive mask of the ViT windows (AI Hub bundle: window_attention_mask range [-1000, 0])
# Additive mask of the LLM. -100 (the InternVL bundle) is not enough for v3: layer 27 has heads whose valid attention
# logits go down to -750 while future tokens of the same prefill chunk score up to +391 above the row maximum, and
# layer 0 reaches +1348; with -100 the last-token logits moved by up to 0.87 vs transformers, with -1e4 by 5e-5.
LLM_MASK = -1e4


@dataclass(frozen=True)
class Spec:
    """Shapes of the deployed model: config.json + the project's static 560x308 VLM input (configs/default.yaml)."""

    image_w: int = 560
    image_h: int = 308
    patch: int = 14
    merge: int = 2
    temporal: int = 2
    window: int = 112
    vit_dim: int = 1280
    vit_heads: int = 16
    vit_depth: int = 32
    vit_full: tuple[int, ...] = (7, 15, 23, 31)
    vit_theta: float = 10000.0
    dim: int = 2048
    layers: int = 36
    heads: int = 16
    kv_heads: int = 2
    mlp: int = 11008
    vocab: int = 151936
    theta: float = 1e6
    mrope: tuple[int, ...] = (16, 24, 24)
    eps: float = 1e-6
    image_token: int = 151655
    eos: tuple[int, ...] = (151645, 151643)

    @property
    def grid_h(self) -> int:
        return self.image_h // self.patch

    @property
    def grid_w(self) -> int:
        return self.image_w // self.patch

    @property
    def patches(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def image_tokens(self) -> int:
        return self.patches // self.merge**2

    @property
    def patch_dim(self) -> int:
        return 3 * self.temporal * self.patch**2

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @classmethod
    def from_model(cls, model_dir: str | Path, image_w: int = 560, image_h: int = 308) -> Spec:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        text, vis = cfg.get("text_config", cfg), cfg["vision_config"]
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        stride = vis["patch_size"] * vis["spatial_merge_size"]
        if image_w % stride or image_h % stride:
            raise ValueError(f"image {image_w}x{image_h} must be a multiple of {stride}")
        return cls(image_w=image_w, image_h=image_h, patch=vis["patch_size"], merge=vis["spatial_merge_size"],
                   temporal=vis["temporal_patch_size"], window=vis["window_size"], vit_dim=vis["hidden_size"],
                   vit_heads=vis["num_heads"], vit_depth=vis["depth"], vit_full=tuple(vis["fullatt_block_indexes"]),
                   vit_theta=float((vis.get("rope_parameters") or {}).get("rope_theta", 10000.0)),
                   dim=text["hidden_size"], layers=text["num_hidden_layers"], heads=text["num_attention_heads"],
                   kv_heads=text["num_key_value_heads"], mlp=text["intermediate_size"], vocab=text["vocab_size"],
                   theta=float(rope.get("rope_theta", text.get("rope_theta", 1e6))),
                   mrope=tuple(rope.get("mrope_section", (16, 24, 24))), eps=text["rms_norm_eps"],
                   image_token=cfg.get("image_token_id") or 151655)


class Checkpoint:
    """Lazy access to the safetensors weights, one tensor at a time (memory-mapped, nothing else is loaded)."""

    ALIASES = (("model.language_model.", "model."), ("model.visual.", "visual."))

    def __init__(self, model_dir: str | Path):
        from safetensors import safe_open

        self._where = {}
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            handle = safe_open(str(f), "pt")
            for key in handle.keys():
                self._where[key] = handle

    def __call__(self, name: str, dtype=torch.float32) -> torch.Tensor:
        key = name
        if key not in self._where:
            for new, old in self.ALIASES:
                if name.startswith(old) and new + name[len(old):] in self._where:
                    key = new + name[len(old):]
                    break
        return self._where[key].get_tensor(key).to(dtype)


# ----------------------------------------------------------------------------------------------- image, host side

def resize_rgb(image, spec: Spec) -> np.ndarray:
    """PIL image or BGR array -> uint8 RGB [H, W, 3] at the fixed input size (PIL bicubic, as reasoning.vlm.to_pil)."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
    image = image.convert("RGB")
    if image.size != (spec.image_w, spec.image_h):
        image = image.resize((spec.image_w, spec.image_h), Image.BICUBIC)
    return np.asarray(image, np.uint8)


def patchify(rgb: np.ndarray, spec: Spec) -> np.ndarray:
    """uint8 RGB [H, W, 3] -> float32 [patches, 1176] in the Qwen2-VL image processor layout: one image is two
    identical frames (temporal patch 2); patches are ordered by 2x2 merge unit, each row as (channel, frame, y, x)."""
    x = (rgb.astype(np.float32) * np.float32(1 / 255) - MEAN) / STD
    x = np.broadcast_to(x.transpose(2, 0, 1)[None], (spec.temporal, 3, spec.image_h, spec.image_w))
    gh, gw, m, p = spec.grid_h, spec.grid_w, spec.merge, spec.patch
    x = x.reshape(spec.temporal, 3, gh // m, m, p, gw // m, m, p).transpose(2, 5, 3, 6, 1, 0, 4, 7)
    return np.ascontiguousarray(x.reshape(gh * gw, spec.patch_dim), np.float32)


def vit_layout(spec: Spec) -> tuple[torch.Tensor, torch.Tensor]:
    """(window_index, window id of each patch in window order) for the fixed grid, as transformers'
    get_vision_window_index for one image: merge units regrouped window by window (windows of 112 px)."""
    lh, lw = spec.grid_h // spec.merge, spec.grid_w // spec.merge
    win = spec.window // spec.merge // spec.patch
    pad_h, pad_w = win - lh % win, win - lw % win
    index = F.pad(torch.arange(lh * lw).reshape(lh, lw), (0, pad_w, 0, pad_h), value=-100)
    nh, nw = (lh + pad_h) // win, (lw + pad_w) // win
    index = index.reshape(nh, win, nw, win).permute(0, 2, 1, 3).reshape(nh * nw, win * win)
    window_of_unit = torch.arange(nh * nw)[:, None].expand(-1, win * win)
    keep = index != -100
    order, window = index[keep], window_of_unit[keep]
    return order, window.repeat_interleave(spec.merge**2)


def vit_inputs(spec: Spec, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """The ViT inputs that depend only on the grid, in window order: 2D rotary cos/sin [P, head_dim/2] (h then w
    angles, as transformers' Qwen2_5_VLVisionRotaryEmbedding) and the additive window / full attention masks."""
    gh, gw, m = spec.grid_h, spec.grid_w, spec.merge
    hpos, wpos = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    block = (gh // m, m, gw // m, m)
    pos = torch.stack([hpos.reshape(block).transpose(1, 2).flatten(), wpos.reshape(block).transpose(1, 2).flatten()], 1)
    order, window = vit_layout(spec)
    pos = pos.reshape(-1, m * m, 2)[order].reshape(-1, 2)
    spatial = spec.vit_dim // spec.vit_heads // 2
    inv = 1.0 / (spec.vit_theta ** (torch.arange(0, spatial, 2, dtype=torch.float) / spatial))
    ang = torch.cat([pos[:, :1].float() * inv, pos[:, 1:].float() * inv], 1)
    same = window[:, None] == window[None, :]
    return {"position_ids_cos": ang.cos().to(dtype), "position_ids_sin": ang.sin().to(dtype),
            "window_attention_mask": torch.where(same, 0.0, VIT_MASK)[None].to(dtype),
            "full_attention_mask": torch.zeros(1, spec.patches, spec.patches, dtype=dtype)}


# ---------------------------------------------------------------------------------------------------- modules

class Conv1x1(nn.Conv2d):
    """Linear layer as a 1x1 convolution on [1, C, 1, N] (the HTP-friendly form); takes and returns [..., N, C]."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        out_c, in_c = weight.shape
        super().__init__(in_c, out_c, 1, bias=bias is not None, dtype=weight.dtype, device=weight.device)
        with torch.no_grad():
            self.weight.copy_(weight[:, :, None, None])
            if bias is not None:
                self.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead, (n, c) = x.shape[:-2], x.shape[-2:]
        y = super().forward(x.reshape(1, n, c).transpose(1, 2).unsqueeze(2))
        return y.squeeze(2).transpose(1, 2).reshape(*lead, n, -1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return y * weight if weight is not None else y


def rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding with cos/sin of half the head dim (same as HF rotate_half with cat(cos, cos))."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class VisionBlock(nn.Module):
    def __init__(self, ck: Checkpoint, i: int, spec: Spec, dtype, vit_rotation: torch.Tensor | None = None):
        super().__init__()
        p = f"visual.blocks.{i}."
        rot = vit_rotation is not None
        g = lambda n: ck(p + n, torch.float64 if rot else dtype)  # noqa: E731
        c = spec.vit_dim
        w = {n: g(f"{n}.weight") for n in ("attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                                         "norm1", "norm2")}
        b = {n: g(f"{n}.bias") for n in ("attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        if rot:  # fold the RMSNorm weights into the readers, rotate the residual (see VisionEncoder)
            q = vit_rotation.double()
            w["attn.qkv"] = (w["attn.qkv"] * w["norm1"]) @ q
            for n in ("mlp.gate_proj", "mlp.up_proj"):
                w[n] = (w[n] * w["norm2"]) @ q
            for n in ("attn.proj", "mlp.down_proj"):
                w[n], b[n] = q.T @ w[n], b[n] @ q
            w["norm1"] = w["norm2"] = None
        w = {k: v.to(dtype) if v is not None else None for k, v in w.items()}
        b = {k: v.to(dtype) for k, v in b.items()}
        qkv_w, qkv_b = w["attn.qkv"], b["attn.qkv"]
        self.q, self.k, self.v = (Conv1x1(qkv_w[j * c:(j + 1) * c], qkv_b[j * c:(j + 1) * c]) for j in range(3))
        self.proj = Conv1x1(w["attn.proj"], b["attn.proj"])
        self.gate = Conv1x1(w["mlp.gate_proj"], b["mlp.gate_proj"])
        self.up = Conv1x1(w["mlp.up_proj"], b["mlp.up_proj"])
        self.down = Conv1x1(w["mlp.down_proj"], b["mlp.down_proj"])
        self.register_buffer("norm1", w["norm1"])
        self.register_buffer("norm2", w["norm2"])
        self.heads, self.eps = spec.vit_heads, spec.eps

    def forward(self, x, cos, sin, mask):
        n, c = x.shape
        hd = c // self.heads
        h = rms_norm(x, self.norm1, self.eps)
        q, k, v = (proj(h).reshape(n, self.heads, hd).transpose(0, 1) for proj in (self.q, self.k, self.v))
        q, k = rope_half(q, cos, sin), rope_half(k, cos, sin)
        a = torch.softmax((q * (1.0 / math.sqrt(hd))) @ k.transpose(1, 2) + mask, -1) @ v
        x = x + self.proj(a.transpose(0, 1).reshape(n, c))
        h = rms_norm(x, self.norm2, self.eps)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class VisionEncoder(nn.Module):
    """ViT + patch merger for the fixed grid: [880, 1176] patches -> [220, 2048] image features (rotated by R when
    the LLM is). Inputs are those of the AI Hub Qwen2.5-VL vision_encoder.bin (cos/sin/masks in window order).

    `vit_rotation` Q (orthogonal, vit_dim x vit_dim) rotates the ViT's own residual stream like R1 does the LLM's:
    patch embedding and the layers writing the residual R^T W, readers W diag(g) Q, the merger's first linear takes
    ln_q's weight and Q per merged patch. Needed for 16-bit activations: the float residual reaches 7.7e3 from
    block 17 and 2.8e4 after block 31 in a few channels while 99.9 % of it stays below ~25-150 (12.8 dB SQNR W8A16)."""

    INPUTS = ("pixel_values", "position_ids_cos", "position_ids_sin", "window_attention_mask", "full_attention_mask")

    def __init__(self, ck: Checkpoint, spec: Spec, rotation: torch.Tensor | None = None, dtype=torch.float32,
                 vit_rotation: torch.Tensor | None = None):
        super().__init__()
        self.spec = spec
        pe = ck("visual.patch_embed.proj.weight", torch.float64).reshape(spec.vit_dim, -1)
        w0, ln_q = ck("visual.merger.mlp.0.weight", torch.float64), ck("visual.merger.ln_q.weight", torch.float64)
        if vit_rotation is not None:
            q = vit_rotation.double()
            pe = q.T @ pe
            unit = spec.merge**2
            w0 = (w0.reshape(w0.shape[0], unit, spec.vit_dim) * ln_q) @ q  # each merged patch: diag(g) Q
            w0, ln_q = w0.reshape(w0.shape[0], -1), None
        self.patch_embed = Conv1x1(pe.to(dtype))
        self.blocks = nn.ModuleList(VisionBlock(ck, i, spec, dtype, vit_rotation) for i in range(spec.vit_depth))
        self.register_buffer("ln_q", ln_q.to(dtype) if ln_q is not None else None)
        self.mlp0 = Conv1x1(w0.to(dtype), ck("visual.merger.mlp.0.bias", dtype))
        w2, b2 = ck("visual.merger.mlp.2.weight"), ck("visual.merger.mlp.2.bias")
        if rotation is not None:
            w2, b2 = rotation.T.double() @ w2.double(), b2.double() @ rotation.double()
        self.mlp2 = Conv1x1(w2.to(dtype), b2.to(dtype))
        order, _ = vit_layout(spec)
        self.register_buffer("order", order)
        self.register_buffer("reverse", torch.argsort(order))

    def forward(self, pixel_values, position_ids_cos, position_ids_sin, window_attention_mask, full_attention_mask):
        s, unit = self.spec, self.spec.merge**2
        x = self.patch_embed(pixel_values)
        x = torch.index_select(x.reshape(-1, unit, s.vit_dim), 0, self.order).reshape(-1, s.vit_dim)
        for i, blk in enumerate(self.blocks):
            mask = full_attention_mask if i in s.vit_full else window_attention_mask
            x = blk(x, position_ids_cos, position_ids_sin, mask)
        x = rms_norm(x, self.ln_q, 1e-6).reshape(-1, unit * s.vit_dim)
        x = self.mlp2(F.gelu(self.mlp0(x)))
        return torch.index_select(x, 0, self.reverse)


def orthogonal(n: int, seed: int = 0) -> torch.Tensor:
    """Orthogonal n x n rotation (float64) that spreads one channel over all: randomized Hadamard for a power of
    two, otherwise Hadamard(2^k) (x) a random orthogonal m x m (n = 2^k m, m odd; 1280 = 256 x 5)."""
    k = n & -n
    if k == n:
        return hadamard(n, seed)
    m = n // k
    g = torch.randn(m, m, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    qm, r = torch.linalg.qr(g)
    qm = qm * torch.sign(torch.diagonal(r))
    return torch.kron(hadamard(k, seed).contiguous(), qm.contiguous())


def hadamard(n: int, seed: int = 0) -> torch.Tensor:
    """Randomized Hadamard rotation D H / sqrt(n) (float64, orthogonal; n a power of two)."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    if h.shape[0] != n:
        raise ValueError(f"no Sylvester Hadamard matrix of size {n}")
    signs = torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed)).double() * 2 - 1
    return signs[:, None] * h / math.sqrt(n)


def layer_weights(ck: Checkpoint, layer: int, rotation: torch.Tensor | None = None, dtype=torch.float32,
                  head_rotation: torch.Tensor | None = None, quantized=None) -> dict:
    """Weights of one decoder layer; with `rotation` the RMSNorm weights are folded in and the residual rotated.
    `head_rotation` H (head_dim x head_dim) also rotates every value head ("R2", folded: v' = v H, so the 8-bit V
    cache sees spread-out channels; o_proj's columns of each query head become B H)."""
    p = f"model.layers.{layer}."
    names = {"q": "self_attn.q_proj.weight", "qb": "self_attn.q_proj.bias", "k": "self_attn.k_proj.weight",
             "kb": "self_attn.k_proj.bias", "v": "self_attn.v_proj.weight", "vb": "self_attn.v_proj.bias",
             "o": "self_attn.o_proj.weight", "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
             "down": "mlp.down_proj.weight", "ln1": "input_layernorm.weight", "ln2": "post_attention_layernorm.weight"}
    w = {k: ck(p + n, torch.float64 if rotation is not None else dtype) for k, n in names.items()}
    if rotation is not None:
        r = rotation.double()
        for n, ln in (("q", "ln1"), ("k", "ln1"), ("v", "ln1"), ("gate", "ln2"), ("up", "ln2")):
            w[n] = (w[n] * w[ln]) @ r
        w["o"], w["down"] = r.T @ w["o"], r.T @ w["down"]
        w["ln1"] = w["ln2"] = None
    if head_rotation is not None:
        hr = head_rotation.to(w["v"].dtype)
        hd = hr.shape[0]
        v, vb = w["v"].reshape(-1, hd, w["v"].shape[1]), w["vb"].reshape(-1, hd)
        w["v"] = (hr.T @ v).reshape(w["v"].shape)
        w["vb"] = (vb @ hr).reshape(-1)
        o = w["o"].reshape(w["o"].shape[0], -1, hd)
        w["o"] = (o @ hr).reshape(w["o"].shape)
    if quantized is not None:  # gptq.QuantizedWeights of this same rotated model: the 4-bit projection weights
        w.update(quantized.layer(layer, torch.float64))
    return {k: v.to(dtype) if v is not None else None for k, v in w.items()}


def head_weights(ck: Checkpoint, rotation: torch.Tensor | None = None, dtype=torch.float32) -> dict:
    """Final norm + LM head (tied to the embedding table); folded and rotated like the layers."""
    e, g = ck("model.embed_tokens.weight"), ck("model.norm.weight")
    try:
        e = ck("lm_head.weight")
    except KeyError:
        pass
    if rotation is None:
        return {"norm": g.to(dtype), "lm_head": e.to(dtype)}
    lm = torch.empty(e.shape, dtype=dtype)
    r = rotation.double()
    for a in range(0, e.shape[0], 16384):  # chunks: the full table in float64 would be 2.5 GB
        lm[a:a + 16384] = ((e[a:a + 16384].double() * g.double()) @ r).to(dtype)
    return {"norm": None, "lm_head": lm}


def embedding_table(ck: Checkpoint, rotation: torch.Tensor | None = None) -> torch.Tensor:
    """The host-side embedding LUT, float32 [vocab, dim] (embed_tokens @ R when rotated)."""
    e = ck("model.embed_tokens.weight")
    if rotation is None:
        return e
    r = rotation.double()
    out = torch.empty_like(e)
    for a in range(0, e.shape[0], 16384):
        out[a:a + 16384] = (e[a:a + 16384].double() @ r).float()
    return out


class DecoderLayer(nn.Module):
    """One LLM layer in the Genie part form: split heads (SHA), 1x1 convs, explicit KV in / new KV out."""

    def __init__(self, w: dict, spec: Spec, head_rotation: torch.Tensor | None = None):
        super().__init__()
        hd = spec.head_dim
        # "R3": with head_rotation H, Q and K are rotated after RoPE (q H . (k H)^T = q . k^T), so the 8-bit K cache
        # holds channels with the outliers spread out; folded nowhere, it is one small matmul per head
        self.register_buffer("r3", None if head_rotation is None else head_rotation.to(w["q"].dtype))
        sl = lambda t, i: t[i * hd:(i + 1) * hd]  # noqa: E731
        self.q = nn.ModuleList(Conv1x1(sl(w["q"], i), sl(w["qb"], i)) for i in range(spec.heads))
        self.k = nn.ModuleList(Conv1x1(sl(w["k"], i), sl(w["kb"], i)) for i in range(spec.kv_heads))
        self.v = nn.ModuleList(Conv1x1(sl(w["v"], i), sl(w["vb"], i)) for i in range(spec.kv_heads))
        self.o = Conv1x1(w["o"])
        self.gate, self.up, self.down = Conv1x1(w["gate"]), Conv1x1(w["up"]), Conv1x1(w["down"])
        self.register_buffer("ln1", w["ln1"])
        self.register_buffer("ln2", w["ln2"])
        self.group, self.eps, self.scale = spec.heads // spec.kv_heads, spec.eps, 1.0 / math.sqrt(hd)

    def forward(self, x, mask, cos, sin, past_k, past_v):
        """x [1, N, C], mask [1, 1, N, CL], cos/sin [1, 1, N, hd/2], past_k [kv, 1, hd, CL-N],
        past_v [kv, 1, CL-N, hd] -> (x, new_k [kv, 1, hd, N], new_v [kv, 1, N, hd])."""
        h = rms_norm(x, self.ln1, self.eps).unsqueeze(1)  # [1, 1, N, C]
        q = [rope_half(p(h), cos, sin) * self.scale for p in self.q]
        k = [rope_half(p(h), cos, sin) for p in self.k]
        if self.r3 is not None:
            q, k = [t @ self.r3 for t in q], [t @ self.r3 for t in k]
        v = [p(h) for p in self.v]
        new_k = torch.cat([t.transpose(2, 3) for t in k], 0)
        new_v = torch.cat(v, 0)
        out = []
        for j in range(len(k)):
            keys = torch.cat([past_k[j:j + 1], new_k[j:j + 1]], 3)
            values = torch.cat([past_v[j:j + 1], new_v[j:j + 1]], 2)
            for qi in q[j * self.group:(j + 1) * self.group]:
                out.append(torch.softmax(qi @ keys + mask, -1) @ values)
        x = x + self.o(torch.cat(out, -1)).squeeze(1)
        h = rms_norm(x, self.ln2, self.eps)
        return x + self.down(F.silu(self.gate(h)) * self.up(h)), new_k, new_v


def part_ranges(spec: Spec, parts: int = 4) -> list[range]:
    per = math.ceil(spec.layers / parts)
    return [range(a, min(a + per, spec.layers)) for a in range(0, spec.layers, per)]


class DecoderPart(nn.Module):
    """Layers `layers` of the LLM (+ final norm and LM head when `last`) with the I/O of one Genie part."""

    def __init__(self, ck: Checkpoint, spec: Spec, layers: range, last: bool, rotation: torch.Tensor | None = None,
                 dtype=torch.float32, head_rotation: torch.Tensor | None = None, quantized=None):
        super().__init__()
        self.spec, self.layer_ids, self.last = spec, list(layers), last
        hr = head_rotation.double() if head_rotation is not None else None
        self.layers = nn.ModuleList(DecoderLayer(layer_weights(ck, i, rotation, dtype, hr, quantized), spec, hr)
                                    for i in layers)
        if last:
            w = head_weights(ck, rotation, dtype)
            self.register_buffer("norm", w["norm"])
            self.lm_head = Conv1x1(w["lm_head"])

    def io_names(self) -> tuple[list[str], list[str]]:
        a, b = self.layer_ids[0], self.layer_ids[-1] + 1
        kv = lambda d: [f"past_{t}_{i}_{d}" for i in self.layer_ids for t in ("key", "value")]  # noqa: E731
        first = "inputs_embeds" if a == 0 else f"hidden_{a}"
        inputs = [first, "attention_mask", "position_ids_cos", "position_ids_sin", *kv("in")]
        return inputs, ["logits" if self.last else f"hidden_{b}", *kv("out")]

    def forward(self, x, attention_mask, position_ids_cos, position_ids_sin, *past):
        new = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, attention_mask, position_ids_cos, position_ids_sin, past[2 * i], past[2 * i + 1])
            new += [k, v]
        if self.last:
            x = self.lm_head(rms_norm(x, self.norm, self.spec.eps))
        return (x, *new)


# --------------------------------------------------------------------------------------- positions and generation

def prompt_positions(n_before: int, n_after: int, spec: Spec) -> torch.Tensor:
    """M-RoPE positions [3, T] of text (n_before tokens, ending with <|vision_start|>) + one image + text, as
    transformers' get_rope_index: text 1D, image (t, row, column) offset by n_before, text after the image resumes
    at n_before + max(rows, columns)."""
    lh, lw = spec.grid_h // spec.merge, spec.grid_w // spec.merge
    rows, cols = torch.meshgrid(torch.arange(lh), torch.arange(lw), indexing="ij")
    image = torch.stack([torch.zeros(lh * lw, dtype=torch.long), rows.flatten(), cols.flatten()]) + n_before
    after = n_before + max(lh, lw) + torch.arange(n_after)
    return torch.cat([torch.arange(n_before).expand(3, -1), image, after.expand(3, -1)], 1)


def mrope_cos_sin(pos3: torch.Tensor, spec: Spec) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [T, head_dim/2] of M-RoPE positions [3, T] (float32 like transformers): frequency i takes its angle
    from the temporal, row or column position by mrope_section (16, 24, 24)."""
    hd = spec.head_dim
    inv = 1.0 / (spec.theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
    ang = pos3[:, :, None].float() * inv  # [3, T, hd/2]
    bounds = np.cumsum((0, *spec.mrope))
    ang = torch.cat([ang[i % 3, :, bounds[i]:bounds[i + 1]] for i in range(len(spec.mrope))], -1)
    return ang.cos(), ang.sin()


class Generator:
    """Runs the decoder parts the way Genie does: the prompt in chunks of `ar` tokens, then one token per step, in a
    KV cache of `cl` positions (new entries appended after the valid ones; free slots masked)."""

    def __init__(self, parts: list[DecoderPart], spec: Spec, cl: int = 1024, ar: int = 128, device=None,
                 dtype=torch.float32, kv_quant=None):
        self.parts, self.spec, self.cl, self.ar = parts, spec, cl, ar
        self.device, self.dtype = device or torch.device("cpu"), dtype
        self.kv_quant = kv_quant  # optional fn(layer, "key"|"value", tensor) -> tensor (simulated cache precision)
        self.on_part = None  # optional fn(part index, ar, inputs, outputs, n valid rows): calibration capture
        self.reset()

    def reset(self) -> None:
        s, hd = self.spec, self.spec.head_dim
        z = lambda *shape: torch.zeros(*shape, dtype=self.dtype, device=self.device)  # noqa: E731
        self.k = [z(s.kv_heads, 1, hd, self.cl) for _ in range(s.layers)]
        self.v = [z(s.kv_heads, 1, self.cl, hd) for _ in range(s.layers)]
        self.n = 0

    @torch.no_grad()
    def _step(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ar: int) -> torch.Tensor:
        """x [n, C] (n <= ar new tokens) -> logits [n, vocab]."""
        n, past = x.shape[0], self.cl - ar
        if self.n + n > past + ar or self.n > past:
            raise ValueError(f"context of {self.cl} positions is full ({self.n} + {n})")
        pad = lambda t: F.pad(t, (0, 0, 0, ar - n))[None, None].to(self.device, self.dtype)  # noqa: E731
        mask = torch.full((ar, self.cl), LLM_MASK, dtype=self.dtype, device=self.device)
        mask[:, :self.n] = 0.0
        mask[:, past:] = torch.triu(torch.full((ar, ar), LLM_MASK, device=self.device), 1)
        mask = mask[None, None]
        h = F.pad(x, (0, 0, 0, ar - n))[None].to(self.device, self.dtype)
        cos, sin = pad(cos), pad(sin)
        for p, part in enumerate(self.parts):
            kv = [t for i in part.layer_ids for t in (self.k[i][..., :past], self.v[i][:, :, :past])]
            out = part(h, mask, cos, sin, *kv)
            if self.on_part is not None:
                self.on_part(p, ar, (h, mask, cos, sin, *kv), out, n)
            h = out[0]
            for j, i in enumerate(part.layer_ids):
                nk, nv = out[1 + 2 * j][..., :n], out[2 + 2 * j][:, :, :n]
                if self.kv_quant is not None:
                    nk, nv = self.kv_quant(i, "key", nk), self.kv_quant(i, "value", nv)
                self.k[i][..., self.n:self.n + n] = nk
                self.v[i][:, :, self.n:self.n + n] = nv
        self.n += n
        return h[0, :n].float()

    def prefill(self, embeds: torch.Tensor, pos3: torch.Tensor) -> torch.Tensor:
        """Prompt embeddings [T, C] + M-RoPE positions [3, T] -> logits of the last prompt token."""
        cos, sin = mrope_cos_sin(pos3, self.spec)
        logits = None
        for a in range(0, embeds.shape[0], self.ar):
            logits = self._step(embeds[a:a + self.ar], cos[a:a + self.ar], sin[a:a + self.ar], self.ar)
        self.next_pos = int(pos3.max()) + 1
        return logits[-1]

    def decode(self, embed: torch.Tensor) -> torch.Tensor:
        """One token embedding [C] at the next position -> its logits."""
        cos, sin = mrope_cos_sin(torch.full((3, 1), self.next_pos), self.spec)
        self.next_pos += 1
        return self._step(embed[None], cos, sin, 1)[0]
