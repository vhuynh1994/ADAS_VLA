"""Monocular geometry: distance, ego-path membership, closing speed, time-to-collision and cut-in detection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import CameraConfig, DepthConfig
from ..types import VEHICLE_CLASSES, Detection, LaneInfo

if TYPE_CHECKING:  # perception.depth imports this module; the runtime import below is lazy
    from .depth import DepthFrame

# Typical real-world sizes (m) used for pinhole distance estimation.
REAL_HEIGHT_M = {
    "person": 1.7, "bicycle": 1.6, "car": 1.5, "motorcycle": 1.5,
    "bus": 3.2, "truck": 3.0, "traffic light": 1.0, "stop sign": 0.75,
}
REAL_WIDTH_M = {"person": 0.5, "bicycle": 0.6, "car": 1.8, "motorcycle": 0.8, "bus": 2.5, "truck": 2.5}
NOT_OBSTACLES = ("traffic light", "stop sign")
MIN_TOP_FRACTION = 0.5  # a clipped close object's visible top is at least this fraction of its class height
CLIP_MARGIN = 0.03  # a box ending within this fraction of the frame height from the bottom edge is "clipped"
BOX_ANCHOR_RANGE_M = (5.0, 80.0)  # pinhole distances of full vehicle boxes usable as depth-alignment anchors


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    return (image_width / 2) / math.tan(math.radians(hfov_deg) / 2)


def ground_distance(y: float, focal_px: float, frame_height: int, cam_height_m: float,
                    horizon_frac: float = 0.5) -> float | None:
    """Flat-road distance of the ground point seen at image row `y` (level camera, horizon at horizon_frac * H)."""
    dy = y - horizon_frac * frame_height
    return focal_px * cam_height_m / dy if dy > 1 else None


def is_clipped(det: Detection, frame_height: int, clip_margin: float = CLIP_MARGIN) -> bool:
    """Does the box touch the bottom edge of the frame (object partly outside the image)?"""
    return det.box[3] >= (1 - clip_margin) * frame_height


def estimate_distance(det: Detection, focal_px: float, frame_height: int | None = None,
                      cam_height_m: float | None = None, clip_margin: float = CLIP_MARGIN,
                      cls_name: str | None = None) -> float | None:
    """Pinhole distance from the box height.

    A box touching the bottom edge of the frame (when `frame_height` is given) belongs to an object that is
    partly outside the image: its height is under-estimated, so the distance would be over-estimated - exactly
    the very close vehicle the safety gate must not miss. Such boxes also use the box width and the ground-plane
    distance of their bottom edge, and keep the smallest estimate (never a larger one) - but only if an object
    that close would still reach `MIN_TOP_FRACTION` of its class height at the box top. Small boxes on the
    dashboard or hood (the frame bottom shows our own car, not the road) fail that test and keep the height
    estimate. `cls_name` overrides the detection's class (a track's consensus class).
    """
    cls_name = cls_name or det.cls_name
    real_h = REAL_HEIGHT_M.get(cls_name)
    x1, y1, x2, y2 = det.box
    h_px = y2 - y1
    if real_h is None or h_px < 2:
        return None
    dist = focal_px * real_h / h_px
    if frame_height is None or not is_clipped(det, frame_height, clip_margin):
        return dist
    close = dist
    real_w = REAL_WIDTH_M.get(cls_name)
    if real_w is not None and x2 - x1 >= 2:
        close = min(close, focal_px * real_w / (x2 - x1))
    if cam_height_m:
        ground = ground_distance(y2, focal_px, frame_height, cam_height_m)
        if ground is not None:
            close = min(close, ground)
        # height of the box top above the road if the object really were `close` metres away
        top_m = cam_height_m + (0.5 * frame_height - y1) * close / focal_px
        if top_m < MIN_TOP_FRACTION * real_h:
            return dist
    return max(close, 1.0)


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
    hist: deque = field(default_factory=deque)  # (t, measured distance) within the velocity window
    gaps: deque = field(default_factory=deque)  # (t, lateral gap to the ego corridor in corridor widths)
    cutting_in: bool = False


def _fit_slope(samples: deque) -> tuple[float, float]:
    """Least-squares slope of (t, value) samples and its standard error."""
    n = len(samples)
    t_mean = sum(t for t, _ in samples) / n
    v_mean = sum(v for _, v in samples) / n
    stt = sum((t - t_mean) ** 2 for t, _ in samples)
    if n < 3 or stt <= 0:
        return 0.0, math.inf
    slope = sum((t - t_mean) * (v - v_mean) for t, v in samples) / stt
    rss = sum((v - v_mean - slope * (t - t_mean)) ** 2 for t, v in samples)
    return slope, math.sqrt(rss / (n - 2) / stt)


class MotionEstimator:
    """Fills distance / position / in-path / closing speed / TTC / cut-in for each detection.

    Closing speed is the least-squares slope of the measured distance over the last `vel_window_s`, reported
    once the samples span `vel_min_span_s` (no TTC for a new track) and only while the fit is consistent: its
    standard error must stay below `vel_max_se_mps` or `vel_max_rel_se` x the speed, otherwise the speed is
    unknown (None). Differencing consecutive frames instead turns the few-percent jitter of a monocular
    distance into tens of m/s at 60 fps (phantom TTCs of ~1.5 s behind a lead at a constant gap). Distances of
    a track use its consensus class (confidence-weighted votes): YOLO flipping a far vehicle between car and
    truck would otherwise double its distance from one frame to the next.

    Cut-in: the lateral gap between a vehicle's box and the ego corridor is measured in corridor widths (so
    perspective does not change it as the vehicle drives on) and compared with its value `cut_in_window_s`
    earlier. A vehicle within `cut_in_max_distance_m` whose gap shrinks faster than `cut_in_rate` and is
    about to overlap the corridor is flagged `cutting_in`; the flag is released at half that rate (hysteresis)
    and drops as soon as the vehicle is in the ego path (it is then the lead object). A vehicle pulling away
    faster than `cut_in_max_pull_away_mps` is not flagged: merging ahead of us while faster than us is not a
    collision threat, and on curves an overtaking car in the next lane can look like it drifts toward the corridor.

    Depth (optional, `depth` config + a `DepthFrame` per update, see perception/depth.py): the per-box value of a
    depth network replaces (`mode: replace`) or caps (`mode: min`) the pinhole distance once the map's scale and
    shift are known from the road anchors and/or full vehicle boxes; a clipped box is never farther than its
    pinhole upper bound either way. Frames without a reliable alignment keep the pinhole distances.
    """

    def __init__(self, camera: CameraConfig, dist_tau_s: float = 0.05, vel_window_s: float = 0.5,
                 vel_min_span_s: float = 0.2, vel_max_se_mps: float = 1.5, vel_max_rel_se: float = 0.3,
                 max_track_age_s: float = 2.0, oncoming_margin_mps: float = 3.0,
                 cut_in_rate: float = 0.25, cut_in_max_distance_m: float = 30.0, cut_in_window_s: float = 0.6,
                 cut_in_max_pull_away_mps: float = 1.0, depth: DepthConfig | None = None):
        from .depth import DepthAligner

        self.camera = camera
        self.depth_cfg = depth
        self._aligner = DepthAligner(depth) if depth is not None else None
        self.dist_tau_s = dist_tau_s
        self.vel_window_s = vel_window_s
        self.vel_min_span_s = vel_min_span_s
        self.vel_max_se_mps = vel_max_se_mps
        self.vel_max_rel_se = vel_max_rel_se
        self.max_track_age_s = max_track_age_s
        self.oncoming_margin_mps = oncoming_margin_mps
        self.cut_in_rate = cut_in_rate
        self.cut_in_max_distance_m = cut_in_max_distance_m
        self.cut_in_window_s = cut_in_window_s
        self.cut_in_max_pull_away_mps = cut_in_max_pull_away_mps
        self._tracks: dict[int, _Track] = {}
        self._votes: dict[int, dict[str, float]] = {}  # track id -> class -> summed confidence

    def reset(self) -> None:
        self._tracks.clear()
        self._votes.clear()
        if self._aligner is not None:
            self._aligner.reset()

    def _track_class(self, det: Detection) -> str:
        if det.track_id is None:
            return det.cls_name
        votes = self._votes.setdefault(det.track_id, {})
        votes[det.cls_name] = votes.get(det.cls_name, 0.0) + det.conf
        return max(votes, key=votes.get)

    def update(self, detections: list[Detection], t: float, lanes: LaneInfo,
               width: int, height: int, ego_speed_mps: float | None = None,
               depth: DepthFrame | None = None) -> None:
        f_px = focal_length_px(width, self.camera.hfov_deg)
        pinhole = [estimate_distance(det, f_px, height, self.camera.mount_height_m, cls_name=self._track_class(det))
                   for det in detections]
        refined = self._depth_distances(detections, pinhole, depth, height, t) if depth is not None else None
        for i, det in enumerate(detections):
            det.distance_m, det.distance_source = pinhole[i], "pinhole"
            if refined is not None and refined[i] is not None:
                det.distance_m, det.distance_source = refined[i], "depth"
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
        for tid in [tid for tid in self._votes if tid not in self._tracks]:
            del self._votes[tid]

    def _depth_distances(self, detections: list[Detection], pinhole: list[float | None], depth: DepthFrame,
                         height: int, t: float) -> list[float | None] | None:
        """Depth-based distance per detection (None where unavailable), fused with the pinhole estimate."""
        cfg = self.depth_cfg
        if cfg is None or self._aligner is None or cfg.mode == "off" or len(depth.stats) != len(detections):
            return None
        if depth.kind == "metric_depth":
            raw = [v if v is not None and v > 0 else None for v in depth.stats]
        else:
            pairs = list(depth.anchors) if "ground" in cfg.anchors else []
            if "boxes" in cfg.anchors:
                lo, hi = BOX_ANCHOR_RANGE_M
                for det, pin, v in zip(detections, pinhole, depth.stats):
                    if (v is not None and pin is not None and lo <= pin <= hi and det.cls_name in VEHICLE_CLASSES
                            and not is_clipped(det, height)):
                        pairs.append((v, 1.0 / pin))
            alignment = self._aligner.update(pairs, t)
            if alignment is None:
                return None
            raw = [alignment.distance(v) for v in depth.stats]
        out: list[float | None] = []
        for det, pin, d in zip(detections, pinhole, raw):
            if d is None:
                out.append(None)
                continue
            if pin is not None and (cfg.mode == "min" or is_clipped(det, height)):
                d = min(d, pin)  # never farther than the pinhole bound of a clipped box (or than pinhole, in `min`)
            out.append(max(d, 1.0))
        return out

    def _update_track(self, det: Detection, t: float, gap: float) -> None:
        if det.track_id is None or det.distance_m is None:
            return
        prev = self._tracks.get(det.track_id)
        if prev is None:
            self._tracks[det.track_id] = _Track(t=t, dist=det.distance_m, hist=deque([(t, det.distance_m)]),
                                                gaps=deque([(t, gap)]))
            return
        dt = t - prev.t
        if dt <= 0:
            return
        a = 1.0 - math.exp(-dt / self.dist_tau_s) if self.dist_tau_s > 0 else 1.0  # time-based EMA
        dist = a * det.distance_m + (1 - a) * prev.dist
        hist = prev.hist
        hist.append((t, det.distance_m))
        while t - hist[0][0] > self.vel_window_s + 1e-9:
            hist.popleft()
        vel = None
        if t - hist[0][0] >= self.vel_min_span_s - 1e-9:
            slope, se = _fit_slope(hist)
            if se <= max(self.vel_max_se_mps, self.vel_max_rel_se * abs(slope)):
                vel = max(-40.0, min(40.0, -slope))

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
                          and dist <= self.cut_in_max_distance_m
                          and (vel is None or vel >= -self.cut_in_max_pull_away_mps))
        det.cutting_in = cutting_in and not det.in_ego_path
        self._tracks[det.track_id] = _Track(t=t, dist=dist, vel=vel, hist=hist, gaps=gaps,
                                            cutting_in=det.cutting_in)

        det.distance_m = dist
        det.closing_speed_mps = vel
        det.ttc_s = dist / vel if vel is not None and vel > 0.5 else None
