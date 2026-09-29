"""Nexar Collision Prediction (Nexar Open Data License, attribution required): real crashes and near-misses.

Each positive video has two human annotations: `time_of_alert` (the hazard becomes visible) and
`time_of_event` (collision / near-miss). Labels derived from them:
  - BRAKE: shortly after the alert and halfway to the event (the hazard is visible and imminent)
  - KEEP:  4 s and 7 s before the alert (normal driving before the hazard appears)
The 1.5 s just before the alert (ambiguous) and everything after the event are not sampled.

Attribution: Moura, Daniel C., and Zvitia, Orly. "Nexar Collison Dataset." Hugging Face, 2025,
https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2

from ..config import Config
from ..sources import FrameHistory
from ..types import EgoState, LatAction, LongAction
from .common import PREV_FRAME_S, RecordWriter, lead_meta, risk_for, template_reason

SCENE_SPEED_KMH = {"Highway": 90.0}  # everything else: urban / suburban speeds
DEFAULT_SPEED_KMH = 45.0
KEEP_OFFSETS_S = (-7.0, -4.0)
BRAKE_REASON = "Hazard entering our path; brake firmly."


def sample_times(alert: float, event: float) -> list[tuple[float, LongAction]]:
    times = [(alert + dt, LongAction.KEEP) for dt in KEEP_OFFSETS_S if alert + dt >= 0.5]
    first = min(alert + 0.2, event - 0.2)
    times.append((first, LongAction.BRAKE))
    mid = (alert + event) / 2
    if event - alert >= 0.8 and mid - first >= 0.2:
        times.append((mid, LongAction.BRAKE))
    return times


def split_for_video(name: str, test_percent: int) -> str:
    bucket = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    return "test_nexar" if bucket < test_percent else "train"


def build(cfg: Config, root: Path, out_dir: Path, test_percent: int = 13, fps_stride: int = 3,
          max_videos: int | None = None) -> int:
    from ..pipeline import ADASPipeline

    cfg.vlm.enabled = False
    pipe = ADASPipeline(cfg, load_vlm=False)
    writer = RecordWriter(out_dir)
    with (root / "train" / "positive" / "metadata.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["time_of_alert"] and r["time_of_event"]]
    rows = rows[:max_videos] if max_videos else rows
    print(f"nexar: {len(rows)} positive videos", flush=True)

    for k, row in enumerate(rows):
        video = root / "train" / "positive" / row["file_name"]
        vid = f"nexar_{Path(row['file_name']).stem}"
        if not video.exists() or writer.has(f"{vid}_s0"):
            continue
        alert, event = float(row["time_of_alert"]), float(row["time_of_event"])
        targets = sample_times(alert, event)
        ego = EgoState(SCENE_SPEED_KMH.get(row["scene"], DEFAULT_SPEED_KMH))
        split = split_for_video(row["file_name"], test_percent)

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start = max(0, int((min(t for t, _ in targets) - 2.0) * fps))  # 2 s of tracking history
        end = int(max(t for t, _ in targets) * fps) + 1
        wanted = {int(round(t * fps)): (t, a) for t, a in targets}
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        pipe.reset()
        history = FrameHistory(PREV_FRAME_S)
        n = 0
        for idx in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            history.push(idx / fps, frame)
            if idx not in wanted and (idx - start) % fps_stride:
                continue
            ctx = pipe.perceive(frame, idx, idx / fps, ego)
            if idx not in wanted:
                continue
            _, long_a = wanted[idx]
            lat_a = LatAction.KEEP_LANE
            target_speed = ego.speed_kmh * (0.4 if long_a is LongAction.BRAKE else 1.0)
            reason = BRAKE_REASON if long_a is LongAction.BRAKE else template_reason(long_a, lat_a, ctx)
            writer.write(f"{vid}_s{n}", frame, {
                "ego_speed_kmh": ego.speed_kmh, "cruise_speed_kmh": ego.speed_kmh,
                "context": ctx.summary_text(),
                "target": {"longitudinal": long_a.value, "lateral": lat_a.value,
                           "target_speed_kmh": round(target_speed), "risk": risk_for(long_a, lat_a).value,
                           "reason": reason},
                "split": split, "source": "nexar", "group": vid, "video": str(video), "frame": idx,
                "label_source": "nexar_annotation", "reviewed": False, "lead": lead_meta(ctx),
                "meta": {"time_of_alert": alert, "time_of_event": event, "t": round(idx / fps, 2),
                         "scene": row["scene"], "light": row["light_conditions"], "weather": row["weather"]},
            }, prev_frame=history.before(idx / fps))
            n += 1
        cap.release()
        if (k + 1) % 25 == 0:
            print(f"  [{k + 1}/{len(rows)}] {writer.count} samples so far", flush=True)
    writer.close()
    return writer.count
