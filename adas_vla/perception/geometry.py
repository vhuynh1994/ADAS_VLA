"""Monocular geometry: distance, ego-path membership, closing speed, time-to-collision and cut-in detection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..config import CameraConfig
from ..types import VEHICLE_CLASSES, Detection, LaneInfo

# Typical real-world sizes (m) used for pinhole distance estimation.
REAL_HEIGHT_M = {
    "person": 1.7, "bicycle": 1.6, "car": 1.5, "motorcycle": 1.5,
    "bus": 3.2, "truck": 3.0, "traffic light": 1.0, "stop sign": 0.75,
}
REAL_WIDTH_M = {"person": 0.5, "bicycle": 0.6, "car": 1.8, "motorcycle": 0.8, "bus": 2.5, "truck": 2.5}
NOT_OBSTACLES = ("traffic light", "stop sign")


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    return (image_width / 2) / math.tan(math.radians(hfov_deg) / 2)


def ground_distance(y: float, focal_px: float, frame_height: int, cam_height_m: float,
                    horizon_frac: float = 0.5) -> float | None:
    """Flat-road distance of the ground point seen at image row `y` (level camera, horizon at horizon_frac * H)."""
    dy = y - horizon_frac * frame_height
    return focal_px * cam_height_m / dy if dy > 1 else None


def estimate_distance(det: Detection, focal_px: float, frame_height: int | None = None,
                      cam_height_m: float | None = None, clip_margin: float = 0.03) -> float | None:
    """Pinhole distance from the box height.

    A box touching the bottom edge of the frame (when `frame_height` is given) belongs to an object that is
    partly outside the image: its height is under-estimated, so the distance would be over-estimated - exactly
    the very close vehicle the safety gate must not miss. Such boxes also use the box width and the ground-plane
    distance of their bottom edge, and keep the smallest estimate (never a larger one).
    """
    real_h = REAL_HEIGHT_M.get(det.cls_name)
    x1, y1, x2, y2 = det.box
    h_px = y2 - y1
    if real_h is None or h_px < 2:
        return None
    dist = focal_px * real_h / h_px
    if frame_height is None or y2 < (1 - clip_margin) * frame_height:
        return dist
    real_w = REAL_WIDTH_M.get(det.cls_name)
    if real_w is not None and x2 - x1 >= 2:
        dist = min(dist, focal_px * real_w / (x2 - x1))
    if cam_height_m:
        ground = ground_distance(y2, focal_px, frame_height, cam_height_m)
        if ground is not None:
            dist = min(dist, ground)
    return max(dist, 1.0)


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
    gaps: deque = field(default_factory=deque)  # (t, lateral gap to the ego corridor in corridor widths)
    cutting_in: bool = False


class MotionEstimator:
    """Fills distance / position / in-path / closing speed / TTC / cut-in for each detection.

    Cut-in: the lateral gap between a vehicle's box and the ego corridor is measured in corridor widths (so
    perspective does not change it as the vehicle drives on) and compared with its value `cut_in_window_s`
    earlier. A vehicle within `cut_in_max_distance_m` whose gap shrinks faster than `cut_in_rate` and is
    about to overlap the corridor is flagged `cutting_in`; the flag is released at half that rate (hysteresis)
    and drops as soon as the vehicle is in the ego path (it is then the lead object).
    """

    def __init__(self, camera: CameraConfig, dist_alpha: float = 0.5, vel_alpha: float = 0.3,
                 max_track_age_s: float = 2.0, oncoming_margin_mps: float = 3.0,
                 cut_in_rate: float = 0.25, cut_in_max_distance_m: float = 30.0, cut_in_window_s: float = 0.6):
        self.camera = camera
        self.dist_alpha = dist_alpha
        self.vel_alpha = vel_alpha
        self.max_track_age_s = max_track_age_s
        self.oncoming_margin_mps = oncoming_margin_mps
        self.cut_in_rate = cut_in_rate
        self.cut_in_max_distance_m = cut_in_max_distance_m
        self.cut_in_window_s = cut_in_window_s
        self._tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, detections: list[Detection], t: float, lanes: LaneInfo,
               width: int, height: int, ego_speed_mps: float | None = None) -> None:
        f_px = focal_length_px(width, self.camera.hfov_deg)
        for det in detections:
            det.distance_m = estimate_distance(det, f_px, height, self.camera.mount_height_m)
            x1, _, x2, y2 = det.box
            left, right = ego_corridor(y2, lanes, width, height)
            overlap = max(0.0, min(x2, right) - max(x1, left))
            det.in_ego_path = overlap >= 0.3 * max(x2 - x1, 1.0) and det.cls_name not in NOT_OBSTACLES
            cx = (x1 + x2) / 2
            det.position = "front-left" if cx < left else "front-right" if cx > right else "ahead"
            # Signed gap between the box and the corridor: > 0 outside, < 0 overlapping (in corridor widths).
            gap = (left - x2 if cx < (left + right) / 2 else x1 - right) / max(right - left, 1.0)
            self._update_track(det, t, gap)
            # Closing faster than we drive means the object itself moves toward us (oncoming traffic).
            if ego_speed_mps is not None and det.closing_speed_mps is not None and det.track_id is not None:
                det.oncoming = det.closing_speed_mps > ego_speed_mps + self.oncoming_margin_mps

        stale = [tid for tid, tr in self._tracks.items() if t - tr.t > self.max_track_age_s]
        for tid in stale:
            del self._tracks[tid]

    def _update_track(self, det: Detection, t: float, gap: float) -> None:
        if det.track_id is None or det.distance_m is None:
            return
        prev = self._tracks.get(det.track_id)
        if prev is None:
            self._tracks[det.track_id] = _Track(t=t, dist=det.distance_m, gaps=deque([(t, gap)]))
            return
        dt = t - prev.t
        if dt <= 0:
            return
        dist = self.dist_alpha * det.distance_m + (1 - self.dist_alpha) * prev.dist
        raw_vel = max(-40.0, min(40.0, (prev.dist - dist) / dt))
        vel = raw_vel if prev.vel is None else self.vel_alpha * raw_vel + (1 - self.vel_alpha) * prev.vel

        gaps = prev.gaps
        gaps.append((t, gap))
        while len(gaps) > 1 and t - gaps[1][0] >= self.cut_in_window_s:
            gaps.popleft()  # keep the oldest sample that is still at least one window old
        cutting_in = prev.cutting_in
        t_old, gap_old = gaps[0]
        if t - t_old >= self.cut_in_window_s / 2:
            rate = (gap - gap_old) / (t - t_old)
            threshold = self.cut_in_rate / 2 if cutting_in else self.cut_in_rate
            cutting_in = (rate <= -threshold and gap <= 0.15 and det.cls_name in VEHICLE_CLASSES
                          and dist <= self.cut_in_max_distance_m)
        det.cutting_in = cutting_in and not det.in_ego_path
        self._tracks[det.track_id] = _Track(t=t, dist=dist, vel=vel, gaps=gaps, cutting_in=det.cutting_in)

        det.distance_m = dist
        det.closing_speed_mps = vel
        det.ttc_s = dist / vel if vel > 0.5 else None
