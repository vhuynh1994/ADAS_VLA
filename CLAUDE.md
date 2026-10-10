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
- Unit tests need no GPU or models: `pip install -e ".[dev]"` then `pytest -q` (100 tests with torch; without it `tests/test_merge.py` is skipped as one item: 96 passed, 1 skipped; must stay green; the same
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

## Changes 2026-09-29 (cloud session: code + tests; validated on the GPU afterwards, see below)
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

### Validation on the owner's PC (2026-09-29)
- AEB hold + hysteresis stretched monocular false positives into seconds of emergency braking (normal following clip:
  AEB 33% of the time). Root causes fixed in perception (`perception/geometry.py`, `detector.py`):
  closing speed was the difference of consecutive distances (tens of m/s of noise at 60 fps) -> least-squares slope
  over `perception.velocity_window_s` (0.5 s), reported only when the fit is consistent; distances use the track's
  consensus class (car<->truck flips doubled them); full-width hood/dashboard box = ego car; the clipped-box
  ground-plane bound only applies when the box top is physically plausible (dashboard ornament was a "person at
  4 m"); overtaking cars pulling away are not cut-ins (`perception.cut_in_max_pull_away_mps`).
- Gate: `safety.aeb_confirm_s: 0.1` + `aeb_confirm_gap_s: 0.1` (AEB acts once its trigger held 0.1 s, counting
  through detector dropouts up to 0.1 s; once active every trigger extends the hold; FCW acts at once). 0.2 s missed
  the real SUV cut-in at 5 m of the crash clip (detector misses it for frames at a time). Golden vectors hold each
  scene for 2 steps (before / after confirmation) + dropout cases, 206 cases; `evaluate.gate_offline` treats a
  sample as persistent (no confirmation). `aeb_min_decel_mps2` and FCW confirmation were tried and dropped (no gain).
- `scripts/gate_replay.py capture|replay`: detector + lanes once on the GPU (`outputs/gate_replay/*.pkl`), then
  geometry + TTC + gate replayed on the CPU per config variant. Set: 111 Nexar test videos (normal driving before
  alert - 1.5 s, hazard alert..event, control window of equal length), 24 reviewed Australian clips, highway sample.
  Pure normal clips: AEB 15.8% of the time / 8.6 per min (cloud code) -> 4.0% / 2.25 (now); AEB on KEEP-labelled AU
  samples 16% -> 1.5%. Trade-off: AEB in 53% of the Nexar hazard windows (control window 25%) vs 76% (35%) before;
  FCW or AEB still in 90% (control 68%).
- Step 2 (`sweep_policy.py --gate`, v3 val): cautious_gated only moves under-braking 8.8% -> 7.0% while over-braking
  14.8% -> 24.2% -> rejected, greedy stays. Eval logs with `probs`: `outputs/eval_v3p_*.jsonl` (and every new eval).
- Step 3: `data/ds_v3` = exactly the ds_v2 samples rebuilt with the current perception (`scripts/v4_data.sh`:
  `build-dataset comma2k19 --only-from data/ds_v2/labels.jsonl`, australian, nexar; ds_v2 review/split overlays
  copied; UK/Udacity dropped, all excluded). GPU checks passed: 2-frame input = 220 visual tokens (586 total, same as
  one image), DataLoader workers fork fine, eval logs probs. `scripts/pipeline_v4.sh`: train v4 (base v3,
  `--brake-weight 2.0 --workers 2`, prev_frame_s 0.5), merge, eval v4 and v3 on ds_v3 (val + Nexar), sweep, demo,
  reports `outputs/report_v4_{val,nexar}.html`. Make v4 the default only if it wins on under-braking first.
- v4 result (2026-09-30, ds_v3 val 900 / Nexar test 424): v3 73.1% joint / under 9.1% / over 14.3%; v4 67.1% /
  6.2% / 23.7% (Nexar: v3 63.7 / 3.5 / 32.8, v4 61.1 / 2.4 / 36.6). v3 with cautious decoding at tau 0.35 on the same
  val gives 67.3% / 6.1% / 23.1% -> v4 only moved along v3's trade-off curve (brake weight 2.0), the 2-frame input
  added no measurable information. v3 stays the default; v4 kept in `models/adas-vlm-v4` (needs prev_frame_s 0.5).

## Changes 2026-10-10 (latency budget + per-stage timing, measured on the owner's PC)
- Budget (step B0 of the owner's deployment notes): `budget:` in `configs/default.yaml` (`BudgetConfig`), table and
  PC numbers in `docs/LATENCY_BUDGET.md` (proposed deadlines, owner to confirm).
- `adas_vla/timing.py`: per-frame `FrameResult.timing` (ms: read, detector = det_pre/model/post/track, lanes =
  lane_pre/model/post, geometry, vlm, gate, control, frame = capture->command without VLM, e2e), `vlm_age_s`
  (timeline, what the gate uses) and `vlm_age_wall_s`; written by `run --log`, summarized at the end of `run` and by
  `adas-vla latency --log x.jsonl [--json]` (nearest-rank p50/p95/p99/max, DMR, longest run of misses).
- YOLOP normalize moved to the GPU (bit-identical lane output on 521 frames): lanes 23.3 -> 15.4 ms p50; the
  remaining ~8 ms is Hough on the full-res mask (CPU), over the 10 ms lanes budget.
- Async VLM thread holds the GIL while waiting for the GPU: every prefill stalls the detector ~240 ms (proved with a
  thread vs process microbenchmark; CUDA side streams / stream priority do not help). New `vlm.mode: process`
  (`_ProcessVLMWorker`, spawn, newest frame sent when the child is idle): no spikes, VLM p50 0.96 s, but perception
  ~2x slower from GPU time-slicing between two contexts (frame p50 50 ms). Sync stays the default for offline runs.
- Lane-only YOLOP 384x640 (`scripts/make_yolop_lane.py`: drops the drivable head, 16:9 input; also writes QAIRT
  calib/eval raws): SA8650P HTP 16.2 -> 4.7 ms p50 (W8A16, O3 + VTCM 8; the old binary used default config, which
  alone cost 34%), W8A16 vs float lane IoU 0.954, board == emulator bit-exact. `YolopLaneDetector` takes square or
  non-square, 2-head or lane-only ONNX. Not the PC default (agreement with the square model: 16:9 IoU 0.86-0.90,
  comma2k19 4:3 only 0.70); select with `--set perception.lane_weights=models/yolop/yolop-lane-384x640.onnx`.
- Board streaming: `board/vla_stream_server.cpp` (QNX 8.0, both context binaries loaded once, JPEG in, boxes + RLE
  lane mask out; protocol in its header) + `adas_vla/perception/board.py` (`perception.backend: board`; ByteTrack and
  lane fitting stay on the PC). Bit-exact vs qnn-net-run only with the SDK's floatToTfN quantization in the LUT (plain
  round(x/scale)-offset differs on ~0.3% of 16-bit inputs). `tests/test_board.py` covers the client without a board.
- Fixed a latent bug: `reset()` in async mode now drops the previous video's VLM result (timeline restarts at 0, so
  the old decision had a negative age and passed the gate's freshness check).

## Changes 2026-10-10 (cloud session: code + tests only, NOT yet run on the GPU)
- `perception/depth.py`: Depth-Anything distance refinement. `DepthModel` (backends `transformers` = HF checkpoint,
  `onnx` = static-shape export such as Qualcomm AI Hub `depth_anything_v2`), `DepthFrame` (per-detection map statistic
  + road anchors, picklable), robust scale/shift fit `1/d = scale*value + shift` on anchors with known distance
  (road rows via `ground_distance`, full vehicle boxes via pinhole), `DepthAligner` (EMA over time, keeps the last
  good fit `align_max_age_s`). Fusion in `MotionEstimator.update(..., depth=DepthFrame)`: `mode: replace` (a clipped
  box is still capped by its pinhole upper bound) | `min` | `off`; frames without a reliable fit keep pinhole.
  `Detection.distance_source` = pinhole | depth; HUD appends `D`. Config `perception.depth.*` (`DepthConfig`,
  **enabled: false** by default); pipeline runs the model in `perceive()` when enabled.
- `scripts/gate_replay.py`: `capture --set perception.depth.enabled=true` stores depth statistics per frame; new
  `depth DATA` subcommand adds them to existing captures (videos re-read, detector not re-run); `replay` uses them
  whenever `perception.depth.mode != off` (`enabled` only controls model loading).
- Tests: `tests/test_depth.py` (16, synthetic maps, CPU only).

### To validate on the owner's PC
1. `python scripts/fetch_hf.py depth-anything/Depth-Anything-V2-Small-hf` (Apache-2.0; ~100 MB), `pytest -q`.
2. `adas-vla run --no-vlm --set perception.depth.enabled=true --source <normal clip> --output outputs/depth.mp4`:
   check the `D` distances on the HUD against the pinhole ones (same clip without the flag) and the perception ms.
3. `python scripts/gate_replay.py depth outputs/gate_replay` then
   `python scripts/gate_replay.py replay outputs/gate_replay --variant pinhole:perception.depth.mode=off
   --variant depth:perception.depth.mode=replace --variant depth_min:perception.depth.mode=min`. Keep depth only if
   `aeb_time_%` / `aeb_per_min` on normal clips drop without `nexar_aeb_%` dropping (control window as reference);
   `min` is the conservative variant. Then sweep `ground_rows`, `anchors` (`[ground]` vs `[ground, boxes]`),
   `max_rel_rmse`, `input_size` (644x364 vs 518x518) the same way; the alignment quality can be inspected by logging
   `MotionEstimator._aligner.alignment` (scale, shift, n, rel_rmse).
4. If it wins: set `perception.depth.enabled: true` in `configs/default.yaml`, rebuild `data/ds_v3` (the stored
   `lead` distances change), re-run `scripts/safety_golden.py --check` (gate unchanged, must stay up to date).
5. Deployment path: export the same checkpoint from AI Hub (`depth_anything_v2`, ONNX) and run it through
   `perception.depth.backend: onnx` (input shape read from the graph; `onnx_normalize` per export) before the HTP step.

## Round 5 plan (2026-10-10, cloud session: prepared, NOT yet run): base model Qwen2.5-VL-7B-Instruct
Owner's decision. Why: the 7B is the Qwen2.5-VL size Qualcomm AI Hub ships optimized (`qwen2_5_vl_7b_instruct`,
Genie w4a16; SA8650P / SA8775P / SA8255P ADP listed) and it is Apache-2.0 (the 3B is Qwen Research, non-commercial).
Same architecture as the 3B: dataset, prompts, `encode_messages`, `LORA_TARGET_REGEX`, 2-frame input and the
first-token `ActionPolicy` carry over; only the fine-tune is redone.
- `configs/vlm_7b.yaml` (base profile; the 7B teacher weights `models/Qwen--Qwen2.5-VL-7B-Instruct` may already be on
  the PC from `teacher_7b.yaml`). `scripts/pipeline_v5_7b.sh`: 20-step smoke (peak VRAM + ETA) -> train 3 epochs with
  the v3 recipe (class share 0.55, lr 2e-4, brake weight 1.0) -> shard-wise merge -> eval v5 vs v3 on ds_v3 (val +
  Nexar; the v3 jsonl from the v4 pipeline are reused) -> sweep -> `run` demo + `explain` -> HTML reports.
- `adas-vla train --optimizer paged_adamw_8bit --max-steps N`: bitsandbytes 8-bit paged optimizer; an N-step run
  prints `peak_mem` and the ETA of the full run, then stops (`training/finetune.py: build_optimizer`).
- `adas-vla merge` now merges **shard by shard** (`training/merge.py`: `W + alpha/r * B @ A` on each safetensors
  shard, keys of the Qwen2.5-VL checkpoint renames resolved, config/tokenizer/processor files copied); the old
  full-model path (16 GB RAM for the 7B in bf16) is `--full`. `tests/test_merge.py` needs torch + safetensors
  (skipped in the minimal CI environment, runs on the PC).
- Expectations / risks: QLoRA 7B peak VRAM ~6.5-7.5 GB on the 8 GB card (the 3B peaked 4.7 GB). If the smoke OOMs:
  `V5_EXTRA="--set vlm.quantize_vision=true"` for ALL 7B stages (train and eval must see the same ViT) and
  `TRAIN_EXTRA="--optimizer paged_adamw_8bit --lora-r 8"`. Step time ~2.5x the 3B (read the ETA of the smoke before
  committing the GPU overnight). PC eval latency ~2x v3 (the <= 1 s target is for the NPU, measure on the board).
  The demo runs `run` then `explain` separately: the 7B VLM and the Qwen3-4B LLM do not fit together in 8 GB.
- Decision rule unchanged: v5 becomes the default (`configs/default.yaml: vlm.model_id`) only if it wins on
  under-braking first on the fixed val split, then on joint accuracy; keep the Nexar crash test as the second table.

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
