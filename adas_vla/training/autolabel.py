"""Bootstrap a training set: run a (preferably larger) teacher VLM + the safety gate over a video.

The label is the *arbitrated* decision, so rule-based safety knowledge (AEB, ACC gap, VRU braking) is
distilled into the student. Labels are marked `needs_review` - review them before training.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from ..config import Config
from ..pipeline import ADASPipeline
from ..sources import iter_frames
from ..types import EgoState


def autolabel(cfg: Config, source: str, out_dir: Path, every: int = 15, max_frames: int | None = None) -> None:
    cfg.vlm.mode = "sync"
    cfg.vlm.every_n_frames = every
    cfg.vlm.max_decision_age_s = 0.0  # label only with the VLM output of that exact frame
    cfg.vlm.trigger = "interval"
    pipe = ADASPipeline(cfg)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"
    ego = EgoState(speed_kmh=cfg.ego_speed_kmh)
    stem = Path(source).stem
    written = 0

    with labels_path.open("a") as f:
        for idx, t, frame in iter_frames(source, max_frames):
            res = pipe.process(frame, idx, t, ego)
            if idx % every != 0:
                continue
            name = f"{stem}_{idx:06d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame)
            vlm_fresh = res.vlm_decision is not None and res.vlm_decision.frame_idx == idx
            rec = {
                "image": f"frames/{name}",
                "ego_speed_kmh": ego.speed_kmh,
                "cruise_speed_kmh": cfg.control.cruise_speed_kmh,
                "context": res.context.summary_text(),
                "target": res.decision.target_dict(),
                "label_source": res.decision.source,
                "teacher": cfg.vlm.model_id if vlm_fresh else None,
                "teacher_raw": pipe.last_raw if vlm_fresh else None,
                "alerts": [a.kind for a in res.alerts],
                "video": source,
                "frame": idx,
                "needs_review": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            t = rec["target"]
            print(f"frame {idx:5d} -> {t['longitudinal']} / {t['lateral']} ({rec['label_source']})", flush=True)
    pipe.close()
    print(f"\nWrote {written} samples to {labels_path}")
