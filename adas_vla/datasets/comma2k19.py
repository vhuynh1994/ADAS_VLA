"""comma2k19 (MIT): highway driving with CAN speed -> labels from what the human driver actually did.

Layout inside Chunk_N.zip:  Chunk_N/<dongle>|<route-time>/<segment>/
    video.hevc                          20 fps, 1164x874
    global_pose/frame_times             per-frame timestamps (s)
    processed_log/CAN/speed/{t,value}   vehicle speed (m/s)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..sources import FrameHistory
from ..types import EgoState, LatAction, LongAction
from .common import PREV_FRAME_S, RecordWriter, lead_meta, risk_for, split_for, template_reason

FPS = 20.0
HFOV_DEG = 65.0  # comma EON road camera: 1164 px wide, focal length ~910 px


def longitudinal_label(speed_at, t: float, horizon: float = 2.0) -> tuple[LongAction, float, float]:
    """Label from the driver's own speed change over the next `horizon` seconds.

    Returns (action, target speed km/h = speed 3 s later, mean acceleration m/s^2).
    """
    v0, v1, v3 = speed_at(t), speed_at(t + horizon), speed_at(t + 3.0)
    accel = (v1 - v0) / horizon
    if v3 < 0.5 and v0 < 8.0:
        action = LongAction.STOP
    elif accel < -2.0:
        action = LongAction.BRAKE
    elif accel < -0.5:
        action = LongAction.DECELERATE
    elif accel > 0.4:
        action = LongAction.ACCELERATE
    else:
        action = LongAction.KEEP
    return action, v3 * 3.6, accel


def detect_lane_changes(offsets: list[tuple[int, float | None]], fps: float = FPS, window_s: float = 1.2,
                        min_jump: float = 0.6, min_gap_s: float = 3.0) -> list[tuple[int, LatAction]]:
    """Find lane-line crossings in the ego lane-offset trace.

    Moving left, the offset runs toward -1 (on the left line); once the car is over the line the detector
    locks onto the new lane and the offset swings positive. Lane smoothing spreads that swing over a few
    frames, so a crossing is a rise of >= min_jump from <= -0.3 to >= +0.05 within window_s (mirrored for right).
    """
    pts = [(i, o) for i, o in offsets if o is not None]
    events: list[tuple[int, LatAction]] = []
    for k, (idx, off) in enumerate(pts):
        action = None
        for j in range(k - 1, -1, -1):
            prev_idx, prev_off = pts[j]
            if (idx - prev_idx) / fps > window_s:
                break
            if prev_off <= -0.3 and off >= 0.05 and off - prev_off >= min_jump:
                action = LatAction.CHANGE_LEFT
            elif prev_off >= 0.3 and off <= -0.05 and prev_off - off >= min_jump:
                action = LatAction.CHANGE_RIGHT
            if action:
                break
        if action and (not events or (idx - events[-1][0]) / fps >= min_gap_s):
            events.append((idx, action))
    return events


def lateral_label(idx: int, events: list[tuple[int, LatAction]], fps: float = FPS,
                  before_s: float = 3.0, after_s: float = 0.5) -> LatAction:
    for event_idx, action in events:
        if event_idx - before_s * fps <= idx <= event_idx + after_s * fps:
            return action
    return LatAction.KEEP_LANE


def list_segments(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        return sorted(n[: -len("video.hevc")] for n in z.namelist() if n.endswith("/video.hevc"))


def pick_evenly(items: list, n: int) -> list:
    if n >= len(items):
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)] if n > 1 else items[:1]


def _load_npy(z: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.load(io.BytesIO(z.read(name)))


def build(cfg: Config, zip_path: Path, out_dir: Path, max_segments: int = 60, every_s: float = 2.0,
          perception_stride: int = 4, val_percent: int = 20, skip_from: list[Path] | None = None,
          only_from: Path | None = None) -> int:
    """skip_from: existing datasets. Their segments are not rebuilt, and routes that are in their val split are
    skipped entirely, so new training data never leaks from a held-out route.
    only_from: rebuild exactly the segments of this dataset (same sample ids, same split per route), e.g. to
    refresh the perception context after a perception change; `max_segments` is ignored."""
    from ..pipeline import ADASPipeline

    cfg.camera.hfov_deg = HFOV_DEG
    cfg.vlm.enabled = False
    pipe = ADASPipeline(cfg, load_vlm=False)
    writer = RecordWriter(out_dir)
    work = zip_path.parent / "extracted"
    work.mkdir(exist_ok=True)
    segments = pick_evenly(list_segments(zip_path), max_segments)
    ref_split: dict[str, str] = {}
    if only_from is not None:
        from ..training.data import load_records

        ref = [r for r in load_records(only_from, include_excluded=True) if r.get("source") == "comma2k19"]
        wanted = {r["video"] for r in ref}
        ref_split = {r["group"]: r["split"] for r in ref}
        segments = [seg for seg in list_segments(zip_path) if seg in wanted]
    done_segments, val_routes = set(), set()
    for other in skip_from or []:
        from ..training.data import load_records, splits_path

        # Val routes must come from the dataset being extended (its splits overlay), not from a copy of it.
        overlay_src = out_dir / "labels.jsonl" if splits_path(out_dir / "labels.jsonl").exists() else other
        val_routes |= {r["group"] for r in load_records(overlay_src, include_excluded=True)
                       if r.get("source") == "comma2k19" and r.get("split") == "val"}
        for r in load_records(other, include_excluded=True):
            if r.get("source") == "comma2k19":
                done_segments.add(r["id"].rsplit("_", 1)[0])
                if r.get("split") == "val":
                    val_routes.add(r["group"])
    sample_step = int(round(every_s * FPS))
    print(f"comma2k19: {len(segments)} segments, one sample every {every_s:.1f} s", flush=True)

    with zipfile.ZipFile(zip_path) as z:
        for k, seg in enumerate(segments):
            route = seg.rstrip("/").split("/")[-2]
            seg_id = f"comma_{route.replace('|', '_')}_{seg.rstrip('/').split('/')[-1]}"
            if writer.has(f"{seg_id}_{0:05d}") or seg_id in done_segments or route in val_routes:
                continue
            try:
                frame_times = _load_npy(z, seg + "global_pose/frame_times")
                speed_t = _load_npy(z, seg + "processed_log/CAN/speed/t")
                speed_v = _load_npy(z, seg + "processed_log/CAN/speed/value").reshape(-1)
            except KeyError as e:
                print(f"  skip {seg}: missing {e}", flush=True)
                continue
            video = work / f"{seg_id}.hevc"
            if not video.exists():
                video.write_bytes(z.read(seg + "video.hevc"))

            def speed_at(t, _t=speed_t, _v=speed_v):
                return float(np.interp(t, _t, _v))

            cruise = max(30.0, round(np.percentile(speed_v, 90) * 3.6 / 5) * 5)
            split = ref_split.get(route) or ("train" if skip_from else split_for(route, val_percent))
            pipe.reset()
            history = FrameHistory(PREV_FRAME_S)
            cap = cv2.VideoCapture(str(video))
            offsets, samples = [], {}
            idx = 0
            while idx < len(frame_times):
                ok, frame = cap.read()
                if not ok:
                    break
                t_abs = float(frame_times[idx])
                t_rel = t_abs - float(frame_times[0])
                history.push(t_rel, frame)
                if idx % perception_stride == 0:
                    ctx = pipe.perceive(frame, idx, t_rel, EgoState(speed_at(t_abs) * 3.6))
                    offsets.append((idx, ctx.lanes.offset_norm))
                    if idx % sample_step == 0 and t_abs + 3.0 <= min(float(frame_times[-1]), float(speed_t[-1])):
                        samples[idx] = (frame, history.before(t_rel), ctx, t_abs)
                idx += 1
            cap.release()
            video.unlink(missing_ok=True)

            events = detect_lane_changes(offsets)
            for i, (frame, prev, ctx, t_abs) in samples.items():
                long_a, target_kmh, accel = longitudinal_label(speed_at, t_abs)
                lat_a = lateral_label(i, events)
                flags = []
                if long_a.rank >= LongAction.DECELERATE.rank and ctx.lead_object() is None:
                    flags.append("slowing without visible lead")  # may be invisible to the camera
                writer.write(f"{seg_id}_{i:05d}", frame, {
                    "ego_speed_kmh": round(ctx.ego.speed_kmh, 1),
                    "cruise_speed_kmh": cruise,
                    "context": ctx.summary_text(),
                    "target": {
                        "longitudinal": long_a.value, "lateral": lat_a.value,
                        "target_speed_kmh": round(target_kmh), "risk": risk_for(long_a, lat_a).value,
                        "reason": template_reason(long_a, lat_a, ctx),
                    },
                    "split": split, "source": "comma2k19", "group": route, "video": seg, "frame": i,
                    "label_source": "can", "reviewed": False, "flags": flags, "lead": lead_meta(ctx),
                    "meta": {"accel_mps2": round(accel, 2), "camera_hfov_deg": HFOV_DEG},
                }, prev_frame=prev)
            print(f"  [{k + 1}/{len(segments)}] {seg_id}: {len(samples)} samples, "
                  f"{len(events)} lane changes, split={split}", flush=True)
    writer.close()
    return writer.count
