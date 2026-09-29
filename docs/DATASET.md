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
