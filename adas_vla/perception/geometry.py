"""Monocular geometry: distance, ego-path membership, closing speed, time-to-collision and cut-in detection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import CameraConfig, Config, DepthConfig
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
HORIZON_RANGE = (0.3, 0.75)  # plausible horizon rows (fraction of the height) of a lane vanishing point
MIN_GROUND_DY = 0.02  # rows below the horizon (fraction of the height) used for ground points at or above it


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


def lane_horizon(lanes: LaneInfo, height: int) -> float | None:
    """Image row where the two lane lines meet (their vanishing point: the horizon of a flat road), if plausible."""
    if not lanes.valid:
        return None
    (m1, b1), (m2, b2) = lanes.left_fit, lanes.right_fit
    if abs(m1 - m2) < 1e-6:
        return None
    y = (b2 - b1) / (m1 - m2)
    return y if HORIZON_RANGE[0] * height <= y <= HORIZON_RANGE[1] * height else None


def lateral_offset_m(x: float, y: float, width: int, height: int, horizon_y: float, cam_height_m: float) -> float:
    """Flat-road lateral distance (m, + = right) of the ground point at image (x, y) from the camera axis.

    X = (x - cx) * Z / f with Z = f * cam_height / (y - horizon): the focal length cancels out.
    """
    dy = max(y - horizon_y, MIN_GROUND_DY * height)
    return (x - width / 2) * cam_height_m / dy


def metric_corridor(y: float, bounds: tuple[float, float], width: int, height: int, horizon_y: float,
                    cam_height_m: float, half_width_m: float) -> tuple[float, float]:
    """`bounds` limited to +-half_width_m around the camera axis at row y (the axis strip if they do not meet)."""
    half = half_width_m * max(y - horizon_y, MIN_GROUND_DY * height) / cam_height_m
    lo, hi = max(bounds[0], width / 2 - half), min(bounds[1], width / 2 + half)
    return (lo, hi) if hi > lo else (width / 2 - half, width / 2 + half)


@dataclass
class _Track:
    t: float
    dist: float
    vel: float | None = None
    hist: deque = field(default_factory=deque)  # (t, measured distance) within the velocity window
    gaps: deque = field(default_factory=deque)  # (t, lateral gap to the ego corridor in corridor widths)
    cutting_in: bool = False
    outer: deque = field(default_factory=deque)  # (t, gap of the box's far edge, None when clipped by the image)
    outer_rate: float | None = None  # its rate over the cut-in window (corridor widths per second)


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

    Corridor width (`corridor_half_width_m`): the lane corridor of a wide road, a lane line taken from the next lane
    or the fallback trapezoid can be several metres wider than our car, which puts parked cars and the next lane's
    traffic in the ego path. The corridor is then capped to +-corridor_half_width_m around the camera axis, using
    flat-road geometry with the horizon at the lane vanishing point (smoothed over `horizon_tau_s`, relaxing to
    camera.horizon_frac while no lanes are seen). `exclude_oncoming`: a vehicle closing faster than the ego speed +
    `oncoming_margin_mps` drives toward us; it is not the lead unless it is cutting in (crossing or turning into
    our path) or its box reaches within `oncoming_head_on_m` of the camera axis (coming at us head-on) - on curves
    and wide roads the oncoming lane otherwise lies inside the corridor. The cap applies to vehicles only, and
    cut-in is still measured against the lane corridor. With `cut_in_oncoming_outer`, a vehicle that does not drive
    along with us (closing faster than the ego speed + `cut_in_outer_from_mps`: parked, crossing or oncoming) only
    counts as cutting in when the far edge of its box comes toward the corridor too (at that fraction of
    `cut_in_rate`; never when the image border clips it): approaching on its own side or parked by the road, its near
    edge grows toward us with the side face that comes into view, which fired AEB at oncoming and parked cars of a
    left-hand-traffic (UK) dashcam video. Vehicles driving with us (merging) keep the near-edge test.

    Depth (optional, `depth` config + a `DepthFrame` per update, see perception/depth.py): the per-box value of a
    depth network replaces (`mode: replace`) or caps (`mode: min`) the pinhole distance once the map's scale and
    shift are known from the road anchors and/or full vehicle boxes; a clipped box is never farther than its
    pinhole upper bound either way. Frames without a reliable alignment keep the pinhole distances.
    """

    def __init__(self, camera: CameraConfig, dist_tau_s: float = 0.05, vel_window_s: float = 0.5,
                 vel_min_span_s: float = 0.2, vel_max_se_mps: float = 1.5, vel_max_rel_se: float = 0.3,
                 max_track_age_s: float = 2.0, oncoming_margin_mps: float = 3.0,
                 cut_in_rate: float = 0.25, cut_in_max_distance_m: float = 30.0, cut_in_window_s: float = 0.6,
                 cut_in_max_pull_away_mps: float = 1.0, depth: DepthConfig | None = None,
                 corridor_half_width_m: float | None = None, exclude_oncoming: bool = False,
                 oncoming_head_on_m: float = 0.5, horizon_tau_s: float = 1.0,
                 cut_in_oncoming_outer: float | None = None, cut_in_outer_from_mps: float = -3.0,
                 oncoming_head_on_max_m: float | None = None):
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
        self.corridor_half_width_m = corridor_half_width_m
        self.exclude_oncoming = exclude_oncoming
        self.oncoming_head_on_m = oncoming_head_on_m
        self.cut_in_oncoming_outer = cut_in_oncoming_outer
        self.cut_in_outer_from_mps = cut_in_outer_from_mps
        self.oncoming_head_on_max_m = oncoming_head_on_max_m
        self.horizon_tau_s = horizon_tau_s
        self._horizon: tuple[float, float] | None = None  # (t, smoothed horizon as a fraction of the height)
        self.horizon_y: float | None = None  # horizon row (px) of the last update(), used by the BEV view
        self._tracks: dict[int, _Track] = {}
        self._votes: dict[int, dict[str, float]] = {}  # track id -> class -> summed confidence

    @classmethod
    def from_config(cls, cfg: Config) -> MotionEstimator:
        p = cfg.perception
        return cls(cfg.camera, cut_in_rate=p.cut_in_rate, cut_in_max_distance_m=p.cut_in_max_distance_m,
                   cut_in_max_pull_away_mps=p.cut_in_max_pull_away_mps, vel_window_s=p.velocity_window_s,
                   depth=p.depth, corridor_half_width_m=p.corridor_half_width_m,
                   exclude_oncoming=p.exclude_oncoming, oncoming_head_on_m=p.oncoming_head_on_m,
                   cut_in_oncoming_outer=p.cut_in_oncoming_outer, cut_in_outer_from_mps=p.cut_in_outer_from_mps,
                   oncoming_head_on_max_m=p.oncoming_head_on_max_m)

    def reset(self) -> None:
        self._tracks.clear()
        self._votes.clear()
        self._horizon = None
        self.horizon_y = None
        if self._aligner is not None:
            self._aligner.reset()

    def _track_class(self, det: Detection) -> str:
        if det.track_id is None:
            return det.cls_name
        votes = self._votes.setdefault(det.track_id, {})
        votes[det.cls_name] = votes.get(det.cls_name, 0.0) + det.conf
        return max(votes, key=votes.get)

    def horizon_row(self, lanes: LaneInfo, t: float, height: int) -> float:
        """Horizon row: the lane vanishing point smoothed over time, relaxing to camera.horizon_frac without lanes."""
        vp = lane_horizon(lanes, height)
        target = vp / height if vp is not None else self.camera.horizon_frac
        if self._horizon is None:
            frac = target
        else:
            dt = max(t - self._horizon[0], 0.0)
            a = 1.0 - math.exp(-dt / self.horizon_tau_s) if self.horizon_tau_s > 0 else 1.0
            frac = a * target + (1 - a) * self._horizon[1]
        self._horizon = (t, frac)
        return frac * height

    def update(self, detections: list[Detection], t: float, lanes: LaneInfo,
               width: int, height: int, ego_speed_mps: float | None = None,
               depth: DepthFrame | None = None) -> None:
        f_px = focal_length_px(width, self.camera.hfov_deg)
        pinhole = [estimate_distance(det, f_px, height, self.camera.mount_height_m, cls_name=self._track_class(det))
                   for det in detections]
        refined = self._depth_distances(detections, pinhole, depth, height, t) if depth is not None else None
        horizon_y = self.horizon_row(lanes, t, height)
        self.horizon_y = horizon_y
        cam_h = self.camera.mount_height_m
        for i, det in enumerate(detections):
            det.distance_m, det.distance_source = pinhole[i], "pinhole"
            if refined is not None and refined[i] is not None:
                det.distance_m, det.distance_source = refined[i], "depth"
            x1, _, x2, y2 = det.box
            lane_left, lane_right = left, right = ego_corridor(y2, lanes, width, height)
            if self.corridor_half_width_m is not None and det.cls_name in VEHICLE_CLASSES:
                # vehicles keep to their lane; pedestrians and cyclists can step into our path within a second
                left, right = metric_corridor(y2, (left, right), width, height, horizon_y, cam_h,
                                              self.corridor_half_width_m)
            overlap = max(0.0, min(x2, right) - max(x1, left))
            # a box wider than the corridor (a truck crossing ahead) is in the path once it covers 30 % of it
            det.in_ego_path = (overlap >= 0.3 * max(min(x2 - x1, right - left), 1.0)
                               and det.cls_name not in NOT_OBSTACLES)
            cx = (x1 + x2) / 2
            det.position = "front-left" if cx < left else "front-right" if cx > right else "ahead"
            # Signed gap between the box and the lane corridor: > 0 outside, < 0 overlapping (in corridor widths).
            on_left, cw = cx < (lane_left + lane_right) / 2, max(lane_right - lane_left, 1.0)
            gap = (lane_left - x2 if on_left else x1 - lane_right) / cw
            # ... and of the box's far edge (None when the image border clips it): it follows a vehicle that moves
            # sideways, but not the side face that grows on the near edge as an oncoming vehicle approaches
            clipped = x1 <= 1 if on_left else x2 >= width - 2
            outer = None if clipped else (lane_left - x1 if on_left else x2 - lane_right) / cw
            self._update_track(det, t, gap, outer)
            # Closing faster than we drive means the object itself moves toward us (oncoming traffic).
            if ego_speed_mps is not None and det.closing_speed_mps is not None and det.track_id is not None:
                det.oncoming = det.closing_speed_mps > ego_speed_mps + self.oncoming_margin_mps
            not_following = (ego_speed_mps is not None and det.closing_speed_mps is not None
                             and det.closing_speed_mps > ego_speed_mps + self.cut_in_outer_from_mps)
            if not_following and det.cutting_in and self.cut_in_oncoming_outer is not None:
                # a parked, crossing or oncoming vehicle only cuts in when its far edge comes toward us too, and once
                # its near edge is inside our lane (on curves the straight lane fits make vehicles that keep to their
                # own lane drift toward the corridor, far edge included)
                tr = self._tracks.get(det.track_id)
                rate = tr.outer_rate if tr is not None else None
                if rate is None or rate > -self.cut_in_oncoming_outer * self.cut_in_rate or gap > 0:
                    det.cutting_in = False
            if self.exclude_oncoming and det.oncoming and not det.cutting_in:
                # a crossing or turning vehicle closes faster than we drive too: only its lateral position tells;
                # beyond oncoming_head_on_max_m a curve puts the oncoming lane in front of the camera axis
                lo, hi = (lateral_offset_m(x, y2, width, height, horizon_y, cam_h) for x in (x1, x2))
                near_axis = min(abs(lo), abs(hi)) <= self.oncoming_head_on_m or lo <= 0 <= hi
                close = self.oncoming_head_on_max_m is None or (det.distance_m or 0.0) <= self.oncoming_head_on_max_m
                if not (near_axis and close):
                    det.in_ego_path = False

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

    def _update_track(self, det: Detection, t: float, gap: float, outer: float | None = None) -> None:
        if det.track_id is None or det.distance_m is None:
            return
        prev = self._tracks.get(det.track_id)
        if prev is None:
            self._tracks[det.track_id] = _Track(t=t, dist=det.distance_m, hist=deque([(t, det.distance_m)]),
                                                gaps=deque([(t, gap)]), outer=deque([(t, outer)]))
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
        outers = prev.outer
        outers.append((t, outer))
        while len(outers) > 1 and t - outers[1][0] >= self.cut_in_window_s:
            outers.popleft()
        (t_o, o_old), outer_rate = outers[0], None
        if outer is not None and o_old is not None and t - t_o >= self.cut_in_window_s / 2:
            outer_rate = (outer - o_old) / (t - t_o)
        self._tracks[det.track_id] = _Track(t=t, dist=dist, vel=vel, hist=hist, gaps=gaps,
                                            cutting_in=det.cutting_in, outer=outers, outer_rate=outer_rate)

        det.distance_m = dist
        det.closing_speed_mps = vel
        det.ttc_s = dist / vel if vel is not None and vel > 0.5 else None
