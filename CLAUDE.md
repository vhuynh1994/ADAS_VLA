# CLAUDE.md — handover notes for Claude sessions

The owner writes in **Vietnamese** (short messages); reply in Vietnamese. Use case: learning / research only,
not commercial. Long-term target: pieces of this stack on a Qualcomm **Snapdragon Ride Elite (SA8797P)**.

## What this is
ADAS **VLA** prototype: camera → perception (YOLO11s + ByteTrack, YOLOP lanes, monocular distance/TTC) →
**VLM** (Qwen2.5-VL-3B, fine-tuned with QLoRA) recommends a meta-action as JSON → **deterministic safety gate**
(ACC, AEB, FCW, VRU braking, speed caps, lane-change confirmation) decides → **P controller** → throttle/brake/steer.
A text **LLM** (Qwen3-4B) explains interventions in Vietnamese. Architecture and commands: `README.md`.

## Environment reality
- Developed on a laptop with an RTX 4060 (8 GB). **Not in git:** `.venv/`, `models/` (~30 GB of weights,
  incl. the fine-tuned `models/adas-vlm-v3`), `data/` (datasets + labels), `checkpoints/`, `outputs/`.
- A cloud session has no GPU and none of those files: edit code, add features, write/run unit tests, update docs.
  Training, evaluation and demos must run on the owner's PC.
- Unit tests need no GPU or models: `pip install -e ".[dev]"` then `pytest -q` (44 tests, must stay green).

## Current status (2026-09-28)
- 3 fine-tune rounds done; default VLM `models/adas-vlm-v3` (warm start from v2, 12,461 training samples:
  comma2k19 CAN labels + Australian near-crash clips + Nexar crashes).
- Held-out val (900 samples, whole routes/clips never trained on): joint accuracy **72.6%**, under-braking **8.8%**,
  valid JSON 100%, latency p50 0.90 s. Nexar crash test: 62.7% / under-braking 4.0% (zero-shot: 47.9%).
- Acceptance targets agreed with the owner: joint ≥ 85%, under-braking ≤ 1%, JSON 100%, latency ≤ 1 s
  → accuracy and under-braking **not met**. Cautious decoding was tried and rejected (≈40% false braking).
- Main error sources: single-frame input (can't see a lead slowing down / an imminent cut-in), missed detections
  of very close vehicles, ambiguous near-crash labels, lane changes too rare (never predicted).

## Suggested next steps (owner decides priority)
1. Multi-frame / video input to the VLM (Qwen2.5-VL supports video) — biggest expected gain on KEEP↔DECELERATE.
2. Perception for very close / cut-in vehicles (bigger detector, fine-tune for the domain).
3. Expert-reviewed val labels (current val labels were reviewed by an AI, see `docs/DATASET.md`).
4. Trajectory action head on the LM hidden state; C++ port of safety gate + controller for the target.

## Rules
- Keep the val split fixed when comparing models (`data/ds_v2/labels.splits.json`); report under-braking first.
- The safety gate must only ever brake *more* than the VLM suggests; keep `tests/test_safety_control.py` passing.
- The repo is **public**: never commit tokens, credentials, datasets, model weights, or content from confidential
  vendor documents. `docs/DEPLOY_SA8797P.md` must stay based on public sources only.
- Dataset licenses: Qwen2.5-VL-3B (Qwen Research), Australian clips (CC-BY-NC-4.0), Nexar (attribution, no
  resale) — fine for research, re-check before any commercial use.
