"""Text-only LLM that explains ADAS interventions from structured data ("why did the system act?").

Text-only LLMs are the most established generative-AI workload on automotive NPUs (a single LM sub-model), and
this one only needs structured inputs that the perception + safety stack already produces.
"""

from __future__ import annotations

import time

from ..config import LLMConfig, resolve_model
from .prompts import LANGUAGE_NAMES

EXPLAIN_SYSTEM_PROMPT = (
    "You are the explanation module of a car's driver-assistance system. You receive structured data "
    "about one ADAS event. Explain to the driver in at most two short sentences what happened and why "
    "the system reacted, then give one short piece of advice if useful. Only mention objects and facts "
    "present in the data; never invent details. Answer in {language}."
)


def event_prompt(event: dict) -> str:
    d = event["decision"]
    # Built outside the f-string: nested same-quote f-strings need Python 3.12 (pyproject supports 3.10+).
    alerts = ", ".join(f"{a['kind']} ({a['level']}): {a['message']}" for a in event["alerts"]) or "none"
    lines = [
        f"Time: {event['t']:.1f} s (frame {event['frame']})",
        f"Alerts: {alerts}",

        f"System action: {d['longitudinal']} / {d['lateral']} (decided by {d['source']}), "
        f"target speed {d['target_speed_kmh']:.0f} km/h",
    ]
    if d.get("reason"):
        lines.append(f"Decision reason: {d['reason']}")
    if event.get("objects"):
        lines.append("Perceived objects:\n" + "\n".join(f"- {o}" for o in event["objects"][:8]))
    return "\n".join(lines)


class TextLLM:
    def __init__(self, cfg: LLMConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .vlm import model_load_kwargs

        self.cfg = cfg
        path = resolve_model(cfg.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, **model_load_kwargs(cfg.quantization, cfg.dtype, cfg.device_map))
        self.model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,  # Qwen3 hybrid models: answer directly; ignored by other templates
        ).to(self.model.device)
        gen_kwargs = {"max_new_tokens": max_new_tokens or self.cfg.max_new_tokens}
        if self.cfg.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def explain(self, event: dict) -> tuple[str, float]:
        language = LANGUAGE_NAMES.get(self.cfg.language, self.cfg.language)
        messages = [
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT.format(language=language)},
            {"role": "user", "content": event_prompt(event) + f"\n\nAnswer in {language}."},
        ]
        t0 = time.perf_counter()
        text = self.generate(messages)
        return text, time.perf_counter() - t0
