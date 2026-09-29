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
- Unit tests need no GPU or models: `pip install -e ".[dev]"` then `pytest -q` (61 tests, must stay green; the same
  suite runs in GitHub Actions on Python 3.10 and 3.12 with only numpy/opencv/pillow/pyyaml/pytest installed).


## Current status (2026-09-28)
- 3 fine-tune rounds done; default VLM `models/adas-vlm-v3` (warm start from v2, 12,461 training samples:
  comma2k19 CAN labels + Australian near-crash clips + Nexar crashes).
- Held-out val (900 samples, whole routes/clips never trained on): joint accuracy **72.6%**, under-braking **8.8%**,
  valid JSON 100%, latency p50 0.90 s. Nexar crash test: 62.7% / under-braking 4.0% (zero-shot: 47.9%).
- Acceptance targets agreed with the owner: joint ≥ 85%, under-braking ≤ 1%, JSON 100%, latency ≤ 1 s
  → accuracy and under-braking **not met**. Cautious decoding was tried and rejected (≈40% false braking).
- Main error sources: single-frame input (can't see a lead slowing down / an imminent cut-in), missed detections
  of very close vehicles, ambiguous near-crash labels, lane changes too rare (never predicted).

## Changes 2026-09-29 (cloud session: code + tests only, NOT yet run on the GPU)
- `reasoning/llm.py`: nested same-quote f-string needed Python 3.12 (pyproject says 3.10+); fixed. CI added
  (`.github/workflows/ci.yml`: pytest 3.10/3.12, compileall, golden-vector check).
- Perception (`perception/geometry.py`): boxes touching the bottom edge use box width + ground plane
  (`camera.mount_height_m`, previously unused) as upper bounds on the distance; adjacent vehicles moving into the ego
  corridor get `Detection.cutting_in` (context text `CUTTING IN from the left/right`, HUD `cut-in`) and count as the
  lead object. Config: `perception.cut_in_rate`, `cut_in_max_distance_m`. YOLOP drivable mask now off by default.
- Safety gate (`control/safety.py`): AEB/FCW are Schmitt triggers with hold time (`safety.aeb_hold_s`, `fcw_hold_s`,
  `hysteresis`); `SafetySupervisor.reset()` is called by `ADASPipeline.reset()`. Golden vectors
  `tests/data/safety_golden.json` (203 cases, `scripts/safety_golden.py`, reader `control/golden.py`) = equivalence
  test for the C++ port. **After an intended gate change: regenerate, review the diff, commit both.**
- VLM: `vlm.prev_frame_s` (0.5 → 2-frame video input, same 220 tokens on Qwen2.5-VL; `sources.FrameHistory`,
  `prompts.visual_content`, `vlm.encode_messages`). `vlm.action_policy: cautious_gated` (escalate only when
  `types.context_has_hazard_cue` is true; `scripts/sweep_policy.py --gate` evaluates it offline from eval JSONL).
- Datasets: builders store `image_prev` (frame 0.5 s earlier) and `lead` (structured lead object) per sample;
  `ds_v2` lacks both (old records fall back to a duplicated frame; no gate replay).
- Training: `--workers` (DataLoader), `--brake-weight` (loss weight of DECELERATE/BRAKE/STOP samples).
- Eval/report: Wilson 95% intervals; `system_under_braking_rate` = under-braking after the safety gate replayed on
  the stored `lead` (`evaluate.gate_offline`), report rows get `final`.

### To validate on the owner's PC (in this order)
1. `pytest -q` (61) and `adas-vla run --no-vlm` on a sample video: cut-in labels and AEB hold must not cause false
   braking; tune `cut_in_rate` / `aeb_hold_s` if they do. Re-run the crash demo (`scripts/pipeline_v3.sh` demo lines).
2. `python scripts/sweep_policy.py outputs/eval_v3.jsonl --data data/ds_v2/labels.jsonl --gate --nexar
   outputs/eval_nexar_v3.jsonl` (no GPU): if held-out under-braking drops without ~40% over-braking, set
   `vlm.action_policy: cautious_gated` with the selected thresholds.
3. 2-frame model: rebuild the datasets into `data/ds_v3` (builders now write `image_prev` + `lead`; keep the val
   routes: `--skip-from data/ds_v2/labels.jsonl` for comma2k19, same `labels.splits.json`), then
   `adas-vla train --data data/ds_v3/labels.jsonl --output checkpoints/lora-v4 --set vlm.prev_frame_s=0.5
   --workers 2 --brake-weight 2.0` (base = `models/adas-vlm-v3`), `merge`, `eval --set vlm.prev_frame_s=0.5`.
   First check on the GPU that `encode_messages` works with `videos=[[PIL, PIL]]` on the installed transformers
   (processor default fps 2 → `second_per_grid_ts` 1.0) and that DataLoader workers fork cleanly with the processor.
4. Compare v4 vs v3 with the intervals in the HTML report, under-braking (VLM alone and after the gate) first.

## Suggested next steps (owner decides priority)
1. Train and evaluate the 2-frame model (v4) as above — biggest expected gain on KEEP↔DECELERATE.
2. Perception for very close / cut-in vehicles beyond the geometric fixes: bigger detector, fine-tune for the domain.
3. Expert-reviewed val labels (current val labels were reviewed by an AI, see `docs/DATASET.md`).
4. Trajectory action head on the LM hidden state; C++ port of safety gate + controller (golden vectors ready).


## Rules
- Keep the val split fixed when comparing models (`data/ds_v2/labels.splits.json`); report under-braking first.
- The safety gate must only ever brake *more* than the VLM suggests; keep `tests/test_safety_control.py` and
  `tests/test_safety_golden.py` passing (regenerate the golden file only for an intended change, review the diff).

- The repo is **public**: never commit tokens, credentials, datasets, model weights, or content from confidential
  vendor documents. `docs/DEPLOY_SA8797P.md` must stay based on public sources only.
- Dataset licenses: Qwen2.5-VL-3B (Qwen Research), Australian clips (CC-BY-NC-4.0), Nexar (attribution, no
  resale) — fine for research, re-check before any commercial use.
