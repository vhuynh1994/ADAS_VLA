"""Dashcam clips without CAN (Australian near-crash set, UK urban, Udacity): teacher VLM + safety gate labels.

These labels are proposals: `reviewed` stays false until a person checks them with `adas-vla review`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import Config
from ..sources import iter_frames, source_fps
from ..types import EgoState
from .common import RecordWriter, split_for, template_reason


@dataclass
class ClipSource:
    name: str
    videos: list[Path]
    ego_speed: Callable[[Path], float]  # km/h per video (no CAN available)
    group: Callable[[Path], str]  # videos in one group share a split
    every_s: float = 1.0


def australian(root: Path) -> ClipSource:
    """qutegocentric/Australian_Roads_Dashcam_Driving (CC-BY-NC-4.0): Crash / Near Crash / Normal clips."""
    meta = {}
    csv_path = root / "video_metadata.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            meta = {row["video_name"]: row for row in csv.DictReader(f)}
    videos = sorted(p for d in ("Near Crash", "Crash", "Normal") for p in (root / d).glob("*.mp4"))

    def speed(p: Path) -> float:
        return 90.0 if meta.get(p.name, {}).get("location") == "Highway" else 45.0

    # Clips come from two YouTube compilations, but each clip is a separate incident filmed by a different
    # dashcam, so each clip is its own group (grouping by compilation would leave only two groups).
    return ClipSource("australian", videos, speed, lambda p: "au_" + p.stem, every_s=1.0)


def uk(root: Path) -> ClipSource:
    """aap9002/UK-Road-DashCam (MIT): urban UK driving, one drive -> one group."""
    return ClipSource("uk", sorted(root.glob("*_FH.MP4")), lambda p: 40.0, lambda p: "uk_" + p.stem[:6],
                      every_s=3.0)


def udacity(root: Path) -> ClipSource:
    return ClipSource("udacity", sorted(root.glob("*.mp4")), lambda p: 90.0, lambda p: "udacity_" + p.stem,
                      every_s=2.0)


PRESETS = {
    "australian": ("data/raw/qutegocentric--Australian_Roads_Dashcam_Driving", australian),
    "uk": ("data/raw/aap9002--UK-Road-DashCam", uk),
    "udacity": ("data/samples", udacity),
}


def build(cfg: Config, source: ClipSource, out_dir: Path, val_percent: int = 20,
          max_frames_per_video: int | None = None, stride: int = 1) -> int:
    """stride > 1 runs perception on every stride-th frame only (sample frames are always processed)."""
    from ..pipeline import ADASPipeline

    cfg.vlm.mode = "sync"
    cfg.vlm.trigger = "interval"
    cfg.vlm.max_decision_age_s = 0.0  # label with the teacher output of that exact frame only
    pipe = ADASPipeline(cfg)
    writer = RecordWriter(out_dir)
    print(f"{source.name}: {len(source.videos)} videos, teacher={cfg.vlm.model_id if cfg.vlm.enabled else 'none'}",
          flush=True)

    for k, video in enumerate(source.videos):
        vid = f"{source.name}_{video.stem.replace(' ', '_')}"
        if writer.has(f"{vid}_{0:05d}"):
            continue
        step = max(1, round(source.every_s * source_fps(str(video))))
        cfg.vlm.every_n_frames = step
        group = source.group(video)
        split = split_for(group, val_percent)
        ego = EgoState(source.ego_speed(video))
        pipe.reset()
        n = 0
        for idx, t, frame in iter_frames(str(video), max_frames_per_video):
            if idx % step and idx % stride:
                continue
            res = pipe.process(frame, idx, t, ego)
            if idx % step:
                continue
            d = res.decision
            target = d.target_dict()
            if d.source == "rules":  # teacher output unusable -> factual template reason
                target["reason"] = template_reason(d.longitudinal, d.lateral, res.context)
            fresh = res.vlm_decision is not None and res.vlm_decision.frame_idx == idx
            writer.write(f"{vid}_{idx:05d}", frame, {
                "ego_speed_kmh": ego.speed_kmh,
                "cruise_speed_kmh": ego.speed_kmh,
                "context": res.context.summary_text(),
                "target": target,
                "split": split, "source": source.name, "group": group, "video": str(video), "frame": idx,
                "label_source": d.source, "teacher": cfg.vlm.model_id if fresh else None,
                "teacher_raw": pipe.last_raw if fresh else None,
                "alerts": [a.kind for a in res.alerts], "reviewed": False,
                "flags": [] if fresh or not cfg.vlm.enabled else ["teacher output unusable"],
            })
            n += 1
        print(f"  [{k + 1}/{len(source.videos)}] {vid}: {n} samples, split={split}", flush=True)
    writer.close()
    return writer.count
