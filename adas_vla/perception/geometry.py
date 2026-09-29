"""Monocular geometry: distance, ego-path membership, closing speed and time-to-collision."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import CameraConfig
from ..types import Detection, LaneInfo

# Typical real-world heights (m) used for pinhole distance estimation.
REAL_HEIGHT_M = {
    "person": 1.7, "bicycle": 1.6, "car": 1.5, "motorcycle": 1.5,
    "bus": 3.2, "truck": 3.0, "traffic light": 1.0, "stop sign": 0.75,
}


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    return (image_width / 2) / math.tan(math.radians(hfov_deg) / 2)


def estimate_distance(det: Detection, focal_px: float) -> float | None:
    real_h = REAL_HEIGHT_M.get(det.cls_name)
    h_px = det.box[3] - det.box[1]
    if real_h is None or h_px < 2:
        return None
    return focal_px * real_h / h_px


def default_corridor(y: float, width: int, height: int) -> tuple[float, float]:
    """Fallback ego corridor (trapezoid) when lane lines are not detected."""
    horizon = 0.55 * height
    t = min(max((y - horizon) / (height - horizon), 0.0), 1.0)
    half = (0.03 + t * (0.30 - 0.03)) * width
    return width / 2 - half, width / 2 + half


def ego_corridor(y: float, lanes: LaneInfo, width: int, height: int) -> tuple[float, float]:
    bounds = lanes.bounds_at(y)
    if bounds is not None and bounds[1] > bounds[0]:
        return bounds
    return default_corridor(y, width, height)


@dataclass
class _Track:
    t: float
    dist: float
    vel: float | None = None


class MotionEstimator:
    """Fills distance / position / in-path / closing speed / TTC for each detection."""

    def __init__(self, camera: CameraConfig, dist_alpha: float = 0.5, vel_alpha: float = 0.3,
                 max_track_age_s: float = 2.0, oncoming_margin_mps: float = 3.0):
        self.camera = camera
        self.dist_alpha = dist_alpha
        self.vel_alpha = vel_alpha
        self.max_track_age_s = max_track_age_s
        self.oncoming_margin_mps = oncoming_margin_mps
        self._tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, detections: list[Detection], t: float, lanes: LaneInfo,
               width: int, height: int, ego_speed_mps: float | None = None) -> None:
        f_px = focal_length_px(width, self.camera.hfov_deg)
        for det in detections:
            det.distance_m = estimate_distance(det, f_px)
            x1, _, x2, y2 = det.box
            left, right = ego_corridor(y2, lanes, width, height)
            overlap = max(0.0, min(x2, right) - max(x1, left))
            det.in_ego_path = overlap >= 0.3 * max(x2 - x1, 1.0) and det.cls_name not in (
                "traffic light", "stop sign")
            cx = (x1 + x2) / 2
            det.position = "front-left" if cx < left else "front-right" if cx > right else "ahead"
            self._update_motion(det, t)
            # Closing faster than we drive means the object itself moves toward us (oncoming traffic).
            if ego_speed_mps is not None and det.closing_speed_mps is not None and det.track_id is not None:
                det.oncoming = det.closing_speed_mps > ego_speed_mps + self.oncoming_margin_mps

        stale = [tid for tid, tr in self._tracks.items() if t - tr.t > self.max_track_age_s]
        for tid in stale:
            del self._tracks[tid]

    def _update_motion(self, det: Detection, t: float) -> None:
        if det.track_id is None or det.distance_m is None:
            return
        prev = self._tracks.get(det.track_id)
        if prev is None:
            self._tracks[det.track_id] = _Track(t=t, dist=det.distance_m)
            return
        dt = t - prev.t
        if dt <= 0:
            return
        dist = self.dist_alpha * det.distance_m + (1 - self.dist_alpha) * prev.dist
        raw_vel = max(-40.0, min(40.0, (prev.dist - dist) / dt))
        vel = raw_vel if prev.vel is None else self.vel_alpha * raw_vel + (1 - self.vel_alpha) * prev.vel
        self._tracks[det.track_id] = _Track(t=t, dist=dist, vel=vel)

        det.distance_m = dist
        det.closing_speed_mps = vel
        det.ttc_s = dist / vel if vel > 0.5 else None
