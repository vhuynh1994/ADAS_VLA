"""Vision-language model wrapper (Hugging Face transformers) for driving decisions and copilot chat."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from PIL import Image

from ..config import VLMConfig, resolve_model
from ..types import DrivingDecision
from .parser import parse_decision
from .prompts import DECISION_PREFIX, chat_messages, decision_messages


def to_pil(image, max_side: int, size: list[int] | None = None) -> Image.Image:
    """Accept a BGR numpy frame or a PIL image.

    With `size=[w, h]` the image is resized exactly (static input shape, as on the NPU);
    otherwise it is downscaled so the longest side is <= max_side.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.convert("RGB")
    if size:
        return image.resize((int(size[0]), int(size[1])), Image.BICUBIC)
    w, h = image.size
    scale = max_side / max(w, h)
    if scale < 1:
        image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    return image


# Vision towers stay in bf16: they are small, and bitsandbytes has no fast kernel for their shapes.
VISION_MODULES = ["visual", "vision_tower", "vision_model", "multi_modal_projector", "lm_head"]


def model_load_kwargs(quantization: str, dtype_name: str, device_map: str, quantize_vision: bool = True) -> dict:
    """from_pretrained kwargs for bf16/fp16 or bitsandbytes 4-bit/8-bit loading."""
    import torch

    dtype = getattr(torch, dtype_name)
    kwargs = {"dtype": dtype, "device_map": device_map}
    if quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=None if quantize_vision else VISION_MODULES,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=None if quantize_vision else VISION_MODULES)
    elif quantization != "none":
        raise ValueError(f"Unsupported quantization: {quantization}")
    return kwargs


REASON_KEY = '"reason"'

LONG_ORDER = ["ACCELERATE", "KEEP", "DECELERATE", "BRAKE", "STOP"]  # by braking strength


def choose_action(probs: dict[str, float], policy: str, tau_decel: float, tau_brake: float) -> str:
    """Greedy argmax, or 'cautious': escalate to BRAKE / DECELERATE when P(at least that much braking) is
    high enough. Never relaxes the greedy choice; STOP is only chosen when it is the most likely action."""
    greedy = max(probs, key=probs.get)
    if policy != "cautious":
        return greedy
    rank = LONG_ORDER.index
    p_ge = lambda a: sum(p for b, p in probs.items() if rank(b) >= rank(a))
    for target, tau in (("BRAKE", tau_brake), ("DECELERATE", tau_decel)):
        if rank(greedy) < rank(target) and p_ge(target) >= tau:
            return target
    return greedy


class ActionPolicy:
    """LogitsProcessor for the first generated token: reads the probability of each longitudinal action
    (one distinct first token per action after DECISION_PREFIX) and forces the chosen one."""

    def __init__(self, tokenizer, prompt_len: int, cfg: VLMConfig):
        self.prompt_len, self.cfg = prompt_len, cfg
        self.first_token = {a: tokenizer(a, add_special_tokens=False).input_ids[0] for a in LONG_ORDER}
        self.probs: dict[str, float] | None = None
        self.choice: str | None = None

    def __call__(self, input_ids, scores):
        import torch

        if input_ids.shape[1] != self.prompt_len:
            return scores
        p = torch.softmax(scores[0].float(), dim=-1)
        raw = {a: float(p[t]) for a, t in self.first_token.items()}
        total = sum(raw.values()) or 1.0
        self.probs = {a: v / total for a, v in raw.items()}
        self.choice = choose_action(self.probs, self.cfg.action_policy, self.cfg.cautious_tau_decel,
                                    self.cfg.cautious_tau_brake)
        forced = torch.full_like(scores, float("-inf"))
        forced[:, self.first_token[self.choice]] = 0.0
        return forced


class _TailTextStop:
    """Stop when any stop string appears in the last few generated tokens.

    Much cheaper than transformers' `stop_strings`, which preprocesses the whole vocabulary on every call.
    """

    def __init__(self, tokenizer, prompt_len: int, stop_strings: list[str], tail: int = 6):
        self.tokenizer, self.prompt_len, self.stop_strings, self.tail = tokenizer, prompt_len, stop_strings, tail

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        tail = input_ids[0, max(self.prompt_len, input_ids.shape[1] - self.tail):]
        text = self.tokenizer.decode(tail, skip_special_tokens=True)
        done = any(s in text for s in self.stop_strings)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def close_before_reason(raw: str) -> str:
    """Turn '{..., "risk": "low", "reason"' into '{..., "risk": "low"}'."""
    cut = raw.find(REASON_KEY)
    if cut == -1:
        return raw
    return raw[:cut].rstrip().rstrip(",").rstrip() + "}"


def append_tokens(inputs, ids):
    """Append token ids to a tokenized prompt, extending every per-token tensor accordingly."""
    import torch

    seq_len = inputs["input_ids"].shape[1]
    for key, value in list(inputs.items()):
        if not torch.is_tensor(value) or value.dim() != 2 or value.shape[1] != seq_len:
            continue
        if key == "input_ids":
            extra = ids.to(value.dtype)
        elif key == "attention_mask":
            extra = torch.ones_like(ids, dtype=value.dtype)
        else:  # e.g. token type ids: text tokens are 0
            extra = torch.zeros_like(ids, dtype=value.dtype)
        inputs[key] = torch.cat([value, extra], dim=1)
    return inputs


def load_model_and_processor(cfg: VLMConfig, for_training: bool = False):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs = model_load_kwargs(cfg.quantization, cfg.dtype, cfg.device_map, cfg.quantize_vision)
    path = resolve_model(cfg.model_id)
    model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    processor = AutoProcessor.from_pretrained(path)
    if cfg.adapter_path and not for_training:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.adapter_path)
    if not for_training:
        model.eval()
    return model, processor


class VisionLanguageModel:
    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self.model, self.processor = load_model_and_processor(cfg)
        self._lock = threading.Lock()  # generate() is not re-entrant

    def generate(self, messages: list[dict], max_new_tokens: int | None = None, prefix: str = "",
                 stop_strings: list[str] | None = None, logits_processor=None) -> str:
        """Generate the assistant reply. `prefix` pre-fills the start of the reply (forced format);
        generation ends early at any of `stop_strings` (the stop string is kept in the output)."""
        import torch

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        if prefix:
            inputs = append_tokens(inputs, self.processor.tokenizer(
                prefix, add_special_tokens=False, return_tensors="pt")["input_ids"])
        # BatchFeature.to() casts only floating tensors (pixel_values), not input_ids.
        inputs = inputs.to(self.model.device, dtype=getattr(torch, self.cfg.dtype))
        gen_kwargs = {"max_new_tokens": max_new_tokens or self.cfg.max_new_tokens}
        if self.cfg.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if logits_processor is not None:
            from transformers import LogitsProcessorList

            logits_processor.prompt_len = inputs["input_ids"].shape[1]
            gen_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
        if stop_strings:
            from transformers import StoppingCriteriaList

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                _TailTextStop(self.processor.tokenizer, inputs["input_ids"].shape[1], stop_strings)])
        with self._lock, torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return prefix + self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def decide(self, image, context_text: str, ego_speed_kmh: float, cruise_speed_kmh: float,
               max_speed_kmh: float = 130.0) -> tuple[DrivingDecision | None, str, float]:
        """Returns (decision or None if unparsable, raw text, latency in seconds)."""
        pil = to_pil(image, self.cfg.image_max_side, self.cfg.image_size)
        messages = decision_messages(pil, context_text, ego_speed_kmh, cruise_speed_kmh, self.cfg.language)
        policy = ActionPolicy(self.processor.tokenizer, 0, self.cfg)
        t0 = time.perf_counter()
        if self.cfg.generate_reason:
            raw = self.generate(messages, prefix=DECISION_PREFIX, logits_processor=policy)
        else:
            # The decision fields come first; stop before the free-text reason (~40% of the tokens).
            raw = close_before_reason(self.generate(messages, prefix=DECISION_PREFIX, stop_strings=[REASON_KEY],
                                                    logits_processor=policy))
        latency = time.perf_counter() - t0
        decision = parse_decision(raw, ego_speed_kmh, max_speed_kmh)
        if decision is not None:
            decision.latency_s = latency
            decision.action_probs = policy.probs
        return decision, raw, latency

    def chat(self, image, context_text: str, ego_speed_kmh: float, history: list[tuple[str, str]],
             question: str) -> str:
        pil = to_pil(image, self.cfg.image_max_side, self.cfg.image_size)
        messages = chat_messages(pil, context_text, ego_speed_kmh, history, question, self.cfg.language)
        return self.generate(messages, max_new_tokens=max(self.cfg.max_new_tokens, 512))
