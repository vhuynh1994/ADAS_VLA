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
- Unit tests need no GPU or models: `pip install -e ".[dev]"` then `pytest -q` (104 tests with torch; without it `tests/test_merge.py` and `tests/test_resume.py` are skipped: 97 passed, 2 skipped; must stay green; the same
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
  added no measurable information. v3 stays the default; v4 adapter in `checkpoints/lora-v4` (needs prev_frame_s 0.5).

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

## YOLO11s W8A16 on the SA8650P HTP: root cause and mixed-precision fix (2026-10-12, PC emulator)
- The 2026-10-05 note "W8A16 breaks in the C2PSA attention" was wrong about the place. Layer-wise SQNR (HTP emulator
  `qnn-net-run --debug` vs ONNX Runtime, `scripts/qnn_layerwise.py`): W8A16 is 21-32 dB through layers 0-7 (W8A8: 7-22)
  and collapses at layer 8 (C3k2, stride 32) in the 16-bit SiLU multiply `/model.8/m.0/cv1/act/Mul` (inputs 25 / 33 dB,
  output -6 dB; it disagrees even with x*sigmoid(x) of its own quantized inputs, corr 0.45). C2PSA only inherits it.
  Forcing that one Mul to FP16 moved the breakage to other stride-32 SiLU multiplies (12x20 tensors): an HTP 16-bit
  elementwise problem at that size, not a single bad encoding. `--restrict_quantization_steps` is not the cause
  (same -6 dB without it; it is only needed because the 16-bit attention MatMul wants a symmetric input, which the
  `attn_softmax_sym.json` overrides give without the global flag).
- Fix: W8A16 with the whole stride-32 path in FP16 (layers 8-10, 20-22, P5 head branch: 131 tensors,
  `scripts/qnn_fp16_overrides.py`; QAIRT 2.46 takes overrides in qairt-converter). Every layer >= 18.9 dB; boxes 37.3 dB,
  scores 28.0 dB (W8A8 22.2 / 15.7). Object-level vs float on the 20 held-out frames: 105/106 found (small 62/63,
  medium 26/26, large 17/17), 0 extra boxes; W8A8 (the binary on the board) 84/106 with large 13/17 and 28 extra boxes.
  The O3 + VTCM 8 MB context binary gives bit-identical outputs to the DLC run on the emulator (40/40). On the board:
  4.88 ms p50 (W8A8 default build 2.16 ms, W8A8 O3 + VTCM 8 2.03 ms); outputs differ slightly from the emulator (FP16 on
  the chip, score <= 0.016) but the board detections are the same: 105/106, 0 extra. Live board pipeline: 33.5 ms p50
  (31.0 with W8A8), 1.30 objects/frame on highway_traffic vs 1.39 for the float PC model (W8A8: 2.02). Now the default
  `DET` of `board/board_server.sh`; boxes are uint16 (0.01 px steps instead of 2.58 px).

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

### Validation on the owner's PC (2026-10-10 evening): depth REJECTED, stays disabled
- Speed: Depth-Anything-V2-Small (transformers, fp32, 644x364) 27.6 ms p50 on the RTX 4060; perception frame
  23.8 -> 50.9 ms p50 on highway_traffic.mp4.
- Distances: on highway_traffic.mp4 depth/pinhole = 0.46 median over 603 matched objects (p10 0.21, p90 0.57) and
  visibly wrong (a small car behind the median barrier: pinhole 65 m, depth 14 m). Likely cause: the road anchors
  assume a level camera (`geometry.ground_distance`, horizon at 0.5 H); the horizon of that clip is at ~0.6 H, so
  every ground anchor is too close. Pinhole (box height) does not depend on the pitch.
- Bug found and fixed: YAML 1.1 reads a bare `off` as False, so `--set perception.depth.mode=off` (the replay
  "pinhole" baseline) silently ran `replace`. `DepthConfig.__post_init__` maps False -> "off" and rejects unknown
  modes (test in tests/test_depth.py). Before the fix the pinhole row equalled the depth row.
- `gate_replay.py depth` on all 136 captures (59,920 frames; backup of the pre-depth captures:
  `outputs/gate_replay.bak_pre_depth`). Replay (normal driving 18.3 min; 111 Nexar, ctrl = chance level):

  | variant | aeb_time_% | aeb_per_min | fcw+aeb_time_% | nexar_aeb_% (ctrl) | nexar_warn_% (ctrl) | au_keep_aeb_% |
  |---|---|---|---|---|---|---|
  | pinhole (mode=off) | 9.24 | 5.08 | 50.16 | 53.15 (25.23) | 90.09 (68.47) | 1.50 |
  | depth (replace) | 5.87 | 3.01 | 57.49 | 34.23 (11.71) | 91.89 (73.87) | 4.50 |
  | depth_min | 6.81 | 3.66 | 59.11 | 40.54 (17.12) | 93.69 (77.48) | 3.00 |
  | anchors [boxes], replace | 9.53 | 5.35 | 52.13 | 54.05 (27.03) | 90.99 (69.37) | 5.00 |
  | anchors [boxes], min | 9.25 | 5.19 | 51.42 | 52.25 (25.23) | 90.99 (69.37) | 3.50 |

  Sweep (`outputs/gate_replay_depth_sweep.txt`): anchors [ground], max_rel_rmse 0.15 / 0.10, boxes + rmse 0.10, both
  modes: every variant that lowers normal-driving AEB lowers Nexar AEB about as much (no better separation from the
  control window); none meets the rule. `ground_rows` and `input_size` cannot be swept in replay (the road anchors are
  computed when the depth statistics are added): they need a new `gate_replay.py depth --force` pass (~1 h GPU each),
  not run. Decision: `perception.depth.enabled` stays false, ds_v3 not rebuilt. If depth is revisited: estimate the
  horizon (lane vanishing point, or a per-source `camera` pitch) before the ground anchors, then re-test.

## Round 5 (2026-10-10 .. 10-13): base model Qwen2.5-VL-7B-Instruct - trained, NOT adopted (v3 stays default)
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
- Run on the owner's PC (2026-10-10 20:32 start): the smoke run with the bf16 ViT went out of memory in backward
  (6.94 GB allocated). With `V5_EXTRA="--set vlm.quantize_vision=true"`, `TRAIN_EXTRA="--optimizer paged_adamw_8bit
  --lora-r 8"` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: peak_mem 7.2 GB, 20.2 s per optimizer step,
  3033 steps (balanced epoch 8088 samples x 3 / grad-accum 8) = 17.0 h. Log: `outputs/v5_7b.log` (first try:
  `outputs/v5_7b_try1_oom.log`).
- 2026-10-11: the first full run reached step 730/3033 (loss 1.13 -> ~0.05) when the laptop lid was closed at 00:40:
  suspend hung the CUDA context (GPU 0 % after resume) and the trainer only saved at the end, so ~4 h were lost.
  Since then: `adas-vla train --save-every N --resume` (`training/finetune.py: save_resume_state /
  load_resume_state`: LoRA tensors, optimizer incl. bitsandbytes 8-bit state, scheduler, epoch order + position,
  python/torch/CUDA RNG, written atomically to `<output>/resume/state.pt` (122 MB for the 7B r8); refused if the run
  settings differ; removed when a run completes; `tests/test_resume.py`). GPU check with the real 7B config: resumed
  at sample 16/8088, identical LR, no extra VRAM (7.2 GB), loss of the next steps within 0.1-0.4 % of an uninterrupted
  run (two uninterrupted runs differ by ~0.03 %: non-deterministic kernels). `pipeline_v5_7b.sh` now trains with
  `--save-every 100 --resume`, skips the smoke / a finished training on a rerun, and re-executes itself under
  `systemd-inhibit --what=sleep:handle-lid-switch` (no password needed for the active session; `NO_INHIBIT=1` to
  opt out). Restarted 2026-10-11 07:26 (`SKIP_SMOKE=1`, same V5_EXTRA / TRAIN_EXTRA / expandable_segments).
- 2026-10-11 15:39: out of memory at step ~1458/3033 (epoch 2): the window held the longest sample of ds_v3
  (`nexar_00693_s1`, 314 text tokens; median 109, p99 213) and backward needed 490 MB with 451 MB free (~0.4 GB of
  headroom for the 7B on 8 GB). The step-1400 checkpoint survived. Training now catches `torch.OutOfMemoryError`
  per sample (`finetune._micro_step`): it drops that accumulation window (gradients may be partial), empties the
  cache, logs the sample id and goes on; the scheduler still steps so resume stays aligned; more than 20 such windows
  abort the run. Resumed 2026-10-12 15:06 from step 1400 (log of the failed part: `outputs/v5_7b_try3_oom1458.log`).
- Decision rule unchanged: v5 becomes the default (`configs/default.yaml: vlm.model_id`) only if it wins on
  under-braking first on the fixed val split, then on joint accuracy; keep the Nexar crash test as the second table.
- Result (2026-10-13 01:15; training ended 00:31 at step 3033, 7 out-of-memory windows dropped in total, final loss
  ~0.02; adapter `checkpoints/lora-v5`, eval with the 4-bit ViT as trained). Paired McNemar tests v3 vs v5:

  | set | metric | v3 | v5 | only v3 / only v5 | p |
  |---|---|---|---|---|---|
  | val (900) | under-braking VLM | 9.1 % | **7.1 %** | 30 / 12 | 0.008 |
  | val | under-braking VLM + gate | 8.0 % | 6.4 % | 25 / 11 | 0.029 |
  | val | over-braking (pred > gt, >= DECELERATE) | **9.1 %** | 12.0 % | 25 / 51 | 0.004 |
  | val | joint accuracy | 73.1 % | 71.1 % | 73 / 55 | 0.13 |
  | Nexar (424) | under-braking VLM | **3.5 %** | 10.1 % | 5 / 33 | <0.001 |
  | Nexar | under-braking VLM + gate | **3.3 %** | 9.2 % | 5 / 30 | <0.001 |
  | Nexar | over-braking | 32.8 % | **24.1 %** | 48 / 11 | <0.001 |
  | Nexar | joint accuracy | 63.7 % | 65.8 % | 44 / 53 | 0.42 |
  | both | latency p50 (PC eval) | **0.87 s** | 1.53 s | | |

  v5 is more cautious on ordinary driving but misses crash frames: 30 Nexar BRAKE samples predicted KEEP (v3: 5).
  Latency 1.5 s is over the 1 s budget; in the live demo (`outputs/demo_crash_v5.mp4`, perception on the same GPU)
  the VLM takes 2.0 s p50 and its decision is older than 2 s in 95 % of frames, so the rule ACC of the gate drives
  almost all the time. Cautious policy on v5 (`sweep_policy.py`, thresholds 0.1/0.1): Nexar under-braking 0.2 % but
  over-braking 51.7 % - unusable. Decision: keep v3. Open questions: v5 with the bf16 ViT (needs > 8 GB to train),
  and the 7B on the board via AI Hub Genie w4a16, where its latency would have to be measured.
- 2026-10-13: merged v1, v2, v4, v5 deleted for disk space (adapters kept in `checkpoints/lora-*`: `adas-vla merge
  --adapter checkpoints/lora-vN --out models/adas-vlm-vN`, base 3B for v1-v4, `--config configs/vlm_7b.yaml` for v5).
- Wasted ViT backward (all rounds so far): `gradient_checkpointing_enable()` calls `enable_input_require_grads()`,
  which also hooks the ViT patch embedding, so every step kept the ViT activations and back-propagated through the
  frozen ViT. `finetune.freeze_vision_autograd` runs the ViT under no_grad (LoRA gradients identical:
  `tests/test_vit_grads.py`). With it the 7B fits with the **bf16 ViT**: 20-step smoke peak 7.1 GB, 19.2 s/step.
- v5b = v5 with the bf16 ViT (question: did the 4-bit ViT cause the Nexar under-braking?): `TAG=v5b SKIP_SMOKE=1
  V5_EXTRA="" TRAIN_EXTRA="--optimizer paged_adamw_8bit --lora-r 8" STOP_AT=07:30
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup scripts/pipeline_v5_7b.sh >> outputs/v5b_7b.log 2>&1 &`,
  ~16.3 h of training -> two nights. Owner's rule: train overnight only. `--stop-at HH:MM` / `STOP_AT` and SIGTERM
  save a resumable checkpoint at the next optimizer step and stop; the pipeline exits without merging until the
  training is complete, the same command continues it.

## Ego corridor in metres + oncoming exclusion (2026-10-12, CPU replay; now the default)
- Cause of the false AEB/FCW on the UK demo (`outputs/uk_p5_base.jsonl`, AEB 35 % of frames): lane corridor median
  7.9 m wide (wide road / a line of the next lane), so parked cars and oncoming traffic were "in path".
- `perception.corridor_half_width_m: 1.2` (half our width + 0.3 m): the in-path test of **vehicles** uses the lane
  corridor capped to +-1.2 m around the camera axis, flat-road geometry with the horizon at the lane vanishing point
  (EMA 1 s; `camera.horizon_frac` 0.5 without lanes; per-clip VP 0.37-0.64 H in the captures, std 0.03 H within a
  clip). Pedestrians / cyclists and the cut-in gap keep the lane corridor. A box wider than the corridor (crossing
  truck) is in path once it covers 30 % of the corridor (was 30 % of the box: missed a stopped truck in AU clip 004).
- `perception.exclude_oncoming: true`: a vehicle closing faster than ego speed + 3 m/s is not the lead unless cutting
  in or its box reaches within `oncoming_head_on_m` 0.5 m of the axis.
- Rejected first versions (measured): cap on everything incl. cut-in and VRUs, and "never brake for oncoming", cut
  normal-driving AEB 9.3 % → 3.8 % but lost real hazards (Nexar AEB 53 % → 37 %, AU brake response 88 % → 59 %):
  turning/crossing cars close faster than we drive (flagged oncoming), cut-ins start 2-3.6 m to the side, crossing
  pedestrians. Requiring 0.3 m intrusion into the metric corridor regained one at-impact AEB but put UK oncoming
  traffic back in path (side faces pull box edges ~0.8 m toward the axis).
- Replay, 136 captures, pinhole (`--variant lane_corridor:perception.corridor_half_width_m=null,perception.exclude_oncoming=false`):
  normal AEB 9.28 → 7.49 % time, 5.08 → 4.75 /min, FCW+AEB 50.7 → 45.7 %; Nexar AEB 53.2 → 51.4 % (ctrl 25.2 →
  18.9), warn 90.1 → 89.2 %; AU keep-AEB 1.5 %, brake response 88.2 % unchanged. The 3 changed hazards: 00457 (base
  FCW on a car 4-11 m to the side, chance), 00865 (braking lead flagged oncoming because the replay assumes 45 km/h;
  FCW still fires 0.2 s before the event), 00930 (base AEB only at the impact frame). UK log replay: AEB 35.4 % →
  15.6 % of frames (13 → 8 onsets).
- Remaining false AEB: cut-in flags of vehicles beside us (perspective makes an overtaken car's box edge drift toward
  the corridor), oncoming buses whose side face reaches the axis, phantom closing speeds (~40 m/s) at 40-50 m. Next:
  cut-in only if the predicted lateral position at TTC is inside the corridor (lateral speed in metres, outer box
  edge - class width for the inner edge), plausibility cap of the closing speed with the CAN ego speed.

- 2026-10-24, UK dashcam `241220_130801_007_FH` (left-hand traffic; the OSD shows 25 MPH, run it with ego 40 km/h):
  9 of 10 false AEB onsets of the board demo were oncoming cars on our right flagged CUTTING IN - the near box edge
  grows toward us with the side face, and on curves the straight lane fits make a car that keeps its lane drift toward
  the corridor, far edge included. Fixes (`perception.cut_in_oncoming_outer` 0.5 / `cut_in_outer_from_mps` -3 /
  `oncoming_head_on_max_m` 30): a vehicle not driving along with us (closing > ego - 3 m/s: parked, crossing, oncoming)
  only cuts in when its far edge also approaches (>= half the cut-in rate, not when clipped by the image border) and
  its near edge is already inside our lane; the head-on exception for oncoming vehicles only within 30 m (farther, a
  curve puts the oncoming lane on our axis). `is_ego_hood` full-width test 0.9 -> 0.85 W (the low UK dashcam's
  dashboard, 88 % wide with its top at 0.49 H, was a "car at 2 m"). Replay (capture with `--video path:60`): UK AEB
  9.7-10.0 -> 1.3 onsets/min, 11.6-12.0 -> 1.3 % of the time, FCW+AEB 36 -> 22 % (17 % at ego 40). 136-clip set
  unchanged where it matters: Nexar AEB in hazard windows 31.5 %, AU brake response 88.2 %; normal-driving AEB 5.03 ->
  4.36 % (2.79 -> 2.29 /min), control AEB 9.0 -> 7.2 %; Nexar warn 91.9 -> 91.0 % = one clip (nexar_00321) whose
  FCW starts 0.9 s before the labelled alert in both runs and now ends one frame before the window. Left: oncoming cars
  passing within ~10 m whose box really overlaps a narrow lane (3 onsets in 3 min).

## VLM on the SA8650P board: InternVL3.5-2B via Genie (2026-10-13)
- Qualcomm AI Hub (`~/workspaces/aihub/.venv`, token in `~/.qai_hub/client.ini`, never in the repo) ships prebuilt Genie
  W4A16 bundles for SA8775P (= HTP v73 / soc 52, loads on the SA8650P with QAIRT 2.46). Qwen2.5-VL-3B is not on AI
  Hub (no release 0.44-0.64); Qwen2.5-VL-7B, Qwen3-VL-4B/8B, InternVL3.5-2B, Gemma-4-E4B are.
- Genie on the board's QNX 8.0 needs: `LD_PRELOAD=/mnt/lib64/libfastrpc_pmem.so` (rpcmem symbols), `board/genie_htp_shim.c`
  (drops HTP context option IO_MEM_ESTIMATION, which the QNX backend rejects), no `vision-param` for the InternVL ViT,
  ViT on NSP1 + LLM on NSP0 (a second engine on the same NSP lands on a second PD, which fails with err 1004), and
  `ulimit -c 0` (QNX writes crash cores to /dev/shmem = RAM). Details and numbers: `board/README.md`.
- InternVL3.5-2B on the board: first token 0.15 s after a 0.59 s ViT (~0.74 s from image to action token), 29.5 tok/s,
  model load 9.2 s once. `board/vlm_genie_demo.py` (dashcam frames -> answers + video). The base model under-brakes
  (KEEP at the cut-in of the AU crash clip): it must be fine-tuned like v3 before it is useful.
- Qwen2.5-VL-7B does not fit: its LLM needs ~4.1 GB on the DSP, one 32-bit PD held 3 of 5 parts (~2.5 GB) and the second
  PD fails. VLMs for this board must stay under ~2.5 GB on the DSP.
- Optimisation (2026-10-13/14, `board/README.md` "Tối ưu VLM", `scripts/bench_board_vlm.py`): 2.50 → 1.89 s per decision
  (stop before "reason" = `fast` config, default; LLM poll=false; uint8 image protocol VLM2). Context 1024 vs 4096: no
  change. The AI Hub bundle already has SHA, Linear→Conv, 8-bit transposed KV returning new entries only, LUT embedding,
  int8 LM head, W4 weights, and a **Hadamard rotation R of the LLM residual space** (`embedding_weights.raw =
  embed_tokens @ R`; the ViT outputs features @ R): any replacement ViT must fold R in (`scripts/export_internvl_vit.py`
  does). Own W8A16 ViT = AI Hub quality, 3 % slower; DLBC no gain on the SA8650P (ViT and YOLOP lane); W8A8 ViT 98.7 ms
  vs 569 ms but broken (-0.4 dB) -> mixed precision is the open lever. Remaining per decision: ViT 0.57 s, prefill
  0.31 s, 22 tokens 0.93 s.
- Quantizing a fine-tuned model with the AI Hub recipe needs a big GPU (SeqMSE/AdaScale refuse CPU; ~40 GB VRAM for 3B):
  LLM/VLM precisions are only w4a16 / w4. On the laptop even the ONNX export of InternVL3.5-2B (CPU, fp32) needs
  > 18 GB of RAM: the OOM killer took it and VS Code twice (2026-10-13). Quantize on a cloud / server GPU machine.

## Round 6 (2026-10-13): InternVL3.5-2B fine-tune - the board model (v5b cancelled)
- Owner's decision: v5b (7B with a bf16 ViT) cancelled, the 7B cannot run on the board; train InternVL3.5-2B instead.
- Trained and evaluated on the transformers-native checkpoint `models/OpenGVLab--InternVL3_5-2B-HF`
  (`configs/vlm_internvl.yaml`: QLoRA nf4 LLM, bf16 InternViT, image resized to 448x448 = one tile = 256 image
  tokens, exactly the board bundle's input; single frame). `scripts/pipeline_internvl.sh` (TAG=ivl1): smoke ->
  train 3 epochs with the v3 recipe (resumable, STOP_AT) -> merge -> eval val + Nexar -> sweep -> demo -> reports vs v3.
- Code: LoRA never touches the vision tower / projector (`finetune.VISION_PARTS`; frozen ViT runs without autograd,
  `tests/test_vit_grads.py` covers Qwen2.5-VL and InternVL). `tests/test_internvl.py` (needs the -HF weights): 256
  image tokens, prompt masking, distinct first tokens of the actions, LoRA modules found in the checkpoint.
- For the board: merge the same adapter into the original remote-code checkpoint `models/OpenGVLab--InternVL3_5-2B`
  (`adas-vla merge --config configs/vlm_internvl.yaml --set vlm.model_id=models/OpenGVLab--InternVL3_5-2B ...`; the
  LLM tensors only differ by name, `merge.RENAMES`), quantize it with the AI Hub recipe (`--checkpoint`, `--skip-veg`:
  the vision encoder is unchanged) on a big-GPU machine, compile for SA8775P, run with board/vlm_genie_demo.py.

## Bird's-eye view, map step 1 (2026-10-19)
- `adas_vla/bev.py` + `--bev` on `run` / `demo` (`BEV=1 scripts/run_demo.sh ...`): panel beside the HUD with the ego
  lane lines and objects on the road plane, same flat-road model as `perception/geometry.py`; the horizon row the
  geometry used is now `SceneContext.horizon_y` (`MotionEstimator.horizon_y`). Lanes drawn to 45 m, objects to 60 m
  (beyond: a marker on the top edge). Monocular estimate: a 0.5 deg pitch error moves a road point at 40 m to 32-55 m.
  Clipped boxes at the side edge (not the bottom) keep their box-height distance, often too far. Demo:
  `outputs/bev_highway.mp4`.
- Map step 2 (2026-10-19): `scripts/build_lane_map.py` (+ `adas_vla/mapping.py`, tests/test_mapping.py): every YOLOP
  lane-mask pixel 4-15 m ahead projected with the BEV model and placed with comma2k19 `global_pose` (quaternions are
  scalar-first, device x forward; camera heading from the quaternion, -0.66 deg from the travel direction). Points
  are stored per metre of camera height, so `--lane-width 3.66` calibrates the height afterwards: the ego lane came out
  3.25 m at 6, 9 and 12 m alike with the assumed 1.22 m (uniform scale, so the horizon is fine) -> height 1.377 m.
  Route 2018-07-27--06-03-57 (I-280 southbound, 13.3 km, 9 min, 4.3 M points, ~75 s on the laptop): marking lines
  30-45 cm wide (FWHM) on 200 m stretches, the adjacent lane line (not used for calibration) 3.68 m away.
  Drive 2018-07-30--13-03-07 with that height: ego lane 3.69-3.70 m. `scripts/compare_lane_maps.py`: lines of the
  two drives over 43 stretches: median 0.64 m apart, mostly a slowly drifting GNSS bias (+0.2..+0.7 m per 4 km);
  after removing a 2 km moving median the residual is 0.20 m median, 0.44 m p90. YOLOP lane masks are continuous
  across dashes (BDD100K polylines), so dash periodicity cannot be used as a check. Pitfall fixed while measuring:
  curvilinear "along" positions must add the offset along the path from the nearest sample (snapping to samples
  1.6 m apart made fake dashes). Next: register drives to each other before fusing (the GNSS bias dominates);
  learned map model (nuScenes-mini has vector maps in `maps/expansion`).

## adas-vlm-v3 on the HTP: own QAIRT build for SA8650P (v73) and SA8797P (v81) (2026-10-23)
Owner's 7-step pipeline (PC run -> verify model / tokenizer / preprocessing / output -> QAIRT/QNN survey -> operators
and quantization -> build for SA8797P + QNX -> run on the board -> latency, RAM, accuracy) on the fine-tuned v3.
No SA8797P board exists: binaries are built for both SoCs (QAIRT 2.46 has hexagon-v81, soc_id 72), run on the SA8650P.
- Code: `adas_vla/deploy/qwen_vl.py` (the model in its HTP form: patchify, ViT with window masks as inputs, LLM in 4
  parts with the I/O of the AI Hub Qwen2.5-VL-7B Genie bundle - SHA, 1x1 convs, transposed 8-bit KV, new KV out,
  M-RoPE cos/sin inputs, LUT embedding - and a Genie-like AR128/AR1 generator), `deploy/qwen_vl_sim.py` (decide() on
  it, float or simulated precisions, plugs into `training.evaluate`), `deploy/gptq.py`, `scripts/v3_qnn.py`
  (check-hf / check-deploy / eval / export / calib-vit / calib / gptq / overrides / eval-part / eval-vit / bundle /
  eval-board / compare), `scripts/v3_qnn_build.sh` (convert / quantize / ctx SOC / info), `tests/test_qwen_vl.py`
  (tiny random Qwen2.5-VL from transformers vs the deployment form, incl. all rotations, GPTQ). Big files in
  `data/qnn_v3` (SSD, ~45 GB): ONNX, calibration raws, DLCs, binaries; docker runs mount the SSD (WORK=).
- Step 2 (fp32, CPU, 3 val prompts vs transformers): token ids, pixel values (4.8e-7), M-RoPE positions identical,
  ViT features cos 1.0, last-token logits max |diff| 5e-5, 24 greedy tokens identical. **The additive LLM mask must
  be deep**: with -100 (InternVL bundle) the logits moved 0.87 - layer 27 has heads whose valid logits reach -748 while
  future tokens of the same prefill chunk score +391 above the row max, layer 0 reaches +1348. Graphs use -1000.
- Float reference (bf16 GPU, plain greedy like Genie) agrees 93-95 % with the PC nf4 evals (`compare`), first 374
  val samples 79.4 % joint (PC nf4 77.5 / 79.1 %). `eval_float.jsonl` + `eval_float_b.jsonl` = the 900.
- Quantization (simulated on 100 val samples, R1 Hadamard on the LLM residual): 4-bit round-to-nearest weights per
  channel 48 % agreement with float (joint 46 vs 74 %), LPBQ-64 RTN 56 %, 8-bit KV alone 98 % -> **the weights**;
  **GPTQ** 4-bit per channel (48 train prompts with their answers, 7 s per layer on the 4060, output error ~10x lower
  than RTN) + 8-bit KV: 98 % agreement, joint 76 %. Per-channel grids go to QAIRT as legacy overrides (LPBQ would
  need the v1.0.0 format), the ONNX holds the GPTQ weights so QAIRT's rounding is exact.
- ViT: Qwen2.5-VL's ViT residual reaches 7.7e3 (block 17) and 2.8e4 (block 31) in a few channels: W8A16 gave 12.8 dB
  SQNR (emulator). Rotating the ViT residual (orthogonal 1280 = H256 (x) Q5, folded into patch embed / readers /
  writers / merger) brings the max to 416 / 1163 and 20.4 dB, token cosine 0.995 (InternVL own ViT: 23.1 dB). **The
  bf16 ViT of the PC pipeline (and of v3's fine-tune) is only 15.2 dB from fp32** (worst token cosine 0.58): never use
  bf16 features as the ViT's float reference (with the residual rotation bf16 only reaches 16.8 dB: the error builds up
  everywhere). Decisions barely care: the float LLM with float32 ViT features (`vit-cache`, `eval --vit-cache`,
  `eval_float_vit32.jsonl`) makes the same decision as with the bf16 ViT on 98.8 % of the 900 (joint 76.9 vs 77.3 %,
  under-braking 10.8 % both), so a ~20 dB ViT on the HTP is not where board errors would come from.
- Simulated board configuration on all 900 val (R1 + GPTQ W4 + KV8, `sim_r1_gptq4_kv8_900.jsonl`): joint 76.1 %,
  under-braking 9.8 %, 96.1 % same decisions as float (77.3 % / 10.8 %); the PC nf4 model is 87-89 % from float.
- QAIRT: every graph converts with standard ops only (Conv 1x1, MatMul, Softmax, RmsNorm fused, Gather in the ViT).
  Emulator, part 1 AR128 vs its float ONNX (same GPTQ weights): hidden_9 18.3 dB, KV 21.7 dB (worst 14.1). Context
  binaries (O3, VTCM 8 MB, AR128 + AR1 weight-shared, `bin/sa8650`, `bin/sa8797` identical layouts): parts 1-3 344 MiB,
  part 4 647 MiB (8-bit LM head; AR128 spill-fill 37 MiB), ViT 658 MiB -> LLM 1.64 GiB on the DSP (< ~2.5 GB per PD).
  `check-bins`: the 75 tensors Genie copies between graphs (KV x 36 layers, hidden 9/18/27) share one encoding each.
  The generator logs 3 "wtshare ... small.is_shared()" errors per part (v73 and v81 alike) and still writes the binary.
- **Board (SA8650P, 2026-10-24)**: Genie loads the own bundle in 13 s (ViT NSP0, LLM NSP1) and its M-RoPE handles our
  11x20 image grid: the first 5 val answers were character-identical to float. `eval-board`, 900 val
  (`eval_board_sa8650.jsonl`): JSON 100 %, joint 75.8 %, under-braking 10.2 % (9.1 % after the gate), 95.6 % same
  decisions as float (simulation said 96.1 %). Timing p50: ViT 307 ms, action token 711 ms, 22 tokens 2.66 s, PC round
  trip 2.70 s (p90 2.84). Memory: CPU RAM 2.59 GB (the 1.24 GB LUT is loaded twice: text encoder + generator), HTP DMA
  2.83 GB (`showmem -s`, server running minus stopped).
- Decode speed (`genie-app` profile of the text dialog, 421-token prompt, log at verbose): Genie runs the AR1 graphs for
  generation (ar1_* in the log); AR1 parts 1-3 take 17.4 ms each (347 MB of W4 weights, ~20 GB/s), part 4 25.4 ms (+ LM
  head ~8 ms) -> 79.6 ms/token, 12.6 tok/s; AR128 parts take ~19 ms, i.e. both are weight-streaming bound. poll=true
  (12.2 tok/s) and NSP1 (12.5) change nothing. AI Hub InternVL measured the same way: 23.6 tok/s, parts of 9 layers ~10 ms
  (~22.6 GB/s): our graphs are within ~10-15 % per byte; v3 is slower because it reads 1.70 GB per token vs 1.01 GB.
  The server's node pipeline adds ~12 ms per token (92 ms). Levers: fewer generated tokens (the action is token 1),
  4-bit LM head, the server overhead, speculative decoding of the JSON.

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
