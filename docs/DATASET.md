# Dataset ds_v2 — how the labels were made

No human reviewed these labels. Instead, quality comes from three layers, and every decision is traceable in
`data/ds_v2/labels.reviews.jsonl` (field `reviewer`).

| Source | License | Label origin | Quality control |
|---|---|---|---|
| comma2k19 (Chunk_1, 100 × 1-min segments, US highway/urban) | MIT | What the human driver actually did (CAN speed 2–3 s ahead; lane changes from the lane-offset trace) | Claude visually reviewed every non-KEEP val sample (201) and all train samples where the driver slowed with no visible lead (57, 9 excluded); the 379 KEEP val samples were accepted after a random 48/48 spot-check |
| Australian Roads Dashcam (108 near-crash/crash/normal clips) | CC-BY-NC-4.0 | Claude labeled each clip as time segments from a filmstrip (e.g. frames 5–8 = BRAKE) | 238+66 samples from non-ego-centric clips excluded via metadata; frames after a collision or with unusable views excluded |
| comma2k19 extra (Chunk_1 rest + Chunk_2, 264 segments) — round 3 | MIT | CAN, as above; train only | Segments of held-out val routes skipped (261 leaked samples caught and excluded); all 264 "slowing without visible lead" samples reviewed visually (75 excluded) |
| Nexar Collision Prediction (750 crash/near-miss videos) — round 3 | Nexar Open Data License (attribution, no resale) | BRAKE at `time_of_alert`+0.2 s and halfway to `time_of_event`; KEEP 4 s and 7 s before the alert | Random spot-check: KEEP 6/6 correct; BRAKE hazard obvious from a single frame in 10/18 (annotators saw the whole video). 13 % of videos held out as `test_nexar` |
| UK Road DashCam, Udacity | MIT | — | Excluded: the 4K UK frames make the detector report a phantom car at ~5 m (dashboard), which would teach the model to ignore close obstacles; Udacity is kept unseen for the demo |

**Split:** by whole route (comma) / whole clip (Australian): val = 900 samples, never seen in training
(`adas-vla split`, stored in `labels.splits.json`).

**Label schema:** `{"longitudinal": KEEP|ACCELERATE|DECELERATE|BRAKE|STOP, "lateral": KEEP_LANE|NUDGE_*|CHANGE_*,
"target_speed_kmh", "risk", "reason"}`; `EMERGENCY_BRAKE` is reserved for the safety gate.

**Known limitations**
- The val labels were reviewed by an AI (Claude), not a domain expert; metrics measure agreement with those labels.
- Australian clips have no CAN, so ego speed is assumed (45 km/h urban, 90 km/h highway) and target speeds are coarse.
- Lateral actions other than KEEP_LANE are rare (≈2 %); lateral accuracy on them is not statistically meaningful.
- CC-BY-NC-4.0 (Australian) is non-commercial — fine for this R&D PoC, not for a product.
- Nexar BRAKE labels mark the moment a hazard *starts*; about 45 % are hard to justify from one frame alone.
- Attribution (Nexar): Moura, Daniel C., and Zvitia, Orly. "Nexar Collison Dataset." Hugging Face, 2025,
  https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction.

**Record fields added 2026-09-29** (written by all builders from now on; `ds_v2` does not have them):
- `image_prev`, `prev_frame_s`: the frame 0.5 s before the sample, for the 2-frame VLM input (`vlm.prev_frame_s`).
  Records without it are trained/evaluated with the current frame twice, which Qwen2.5-VL encodes like one image.
- `lead`: the nearest object in the ego path at that frame (`cls`, `distance_m`, `closing_speed_mps`, `ttc_s`,
  `cutting_in`) or `null`. `adas-vla eval` replays the safety gate on it and reports under-braking *after* the gate
  (the system) next to the VLM-only number. Rebuild the dataset to get these numbers for `ds_v2` routes.

## Public labelled datasets on disk (downloaded 2026-10-10, not yet used for training)

Not in git (`data/` is ignored). Research / non-commercial use only, see each license.

| Dataset | Where (`data/raw/…`) | What is there | License |
|---|---|---|---|
| BDD100K (Berkeley DeepDrive) | `bdd100k` → ADAS_CNN `data/bdd100k` | 100k dashcam frames 1280×720 (70k train / 10k val / 20k test, US, 40 % night); 2018 labels per frame: 2D boxes (car, person, rider, bike, motor, truck, bus, traffic light/sign), **lane polylines** (single/double white/yellow, road curb, crosswalk), drivable / alternative area polygons, weather / scene / time of day; drivable masks (`extra/drivable`, 80k); semantic segmentation (`extra/sem_seg` + `extra/images_10k`, 8k); **GPS (speed, course) 1 Hz + accelerometer / gyro ~50 Hz per 40 s video** (`raw/bdd100k_info.zip`, 100k JSON, 59.6 GB unpacked: read members from the zip) | BAIR: educational, research, not-for-profit (`LICENSE.txt`) |
| DriveLM-nuScenes | `OpenDriveLab--DriveLM` (SSD) | 696 scenes / 4,072 key frames × 6 cameras (`nuscenes/samples`), 377,956 QA pairs: perception 162k, prediction 124k, planning 88k, behavior 4k (`v1_1_train_nus.json`); val 799 key frames, questions only (`v1_1_val_nus_q_only.json`, `val_data/`) | CC BY-NC-SA 4.0 (gated on Hugging Face; images from nuScenes) |
| DriveLM-CARLA | same | 187k QA files for CARLA leaderboard-2 key frames (`drivelm_carla_vqas/`, `drivelm_carla_keyframes.txt`); sensor data not downloaded | CC BY-NC-SA 4.0 |

Notes
- The public BDD100K mirror's `bdd100k_det_20_labels.zip` holds 2,000 tracking-style frames, not the 2020 detection
  labels; the 2018 labels already have boxes for all 100k frames. Official 2020 lane / detection packages need a login
  on the BDD100K site.
- BDD100K GPS speed lets the comma2k19 recipe (label = what the driver did next) run on 100k more frames.
- nuScenes and Waymo Open need the owner's own registration / license acceptance; not downloaded.
