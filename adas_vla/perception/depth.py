"""Monocular depth network (Depth-Anything) refining the pinhole distances.

The pinhole estimate (box height x known class height) is the main source of distance noise in this pipeline: a
few percent of jitter per frame, a factor of two when YOLO flips car <-> truck, over-estimates for boxes cut by
the frame edge. A dense depth network sees the object itself. Depth-Anything V2 (the checkpoints Qualcomm AI Hub
ships as `depth_anything_v2`) outputs *relative* inverse depth, `1 / d = scale * value + shift`, with an unknown
scale and shift per image. Both are recovered every frame from anchors whose metric distance we already trust from
the camera geometry: road pixels in front of the car (`ground_distance`: flat road + camera mount height) and,
optionally, full (not clipped) vehicle boxes with their pinhole distance. The fit is a robust least squares,
smoothed over time; when it is not reliable the pinhole distance is kept. Metric checkpoints
(Depth-Anything-V2-Metric-Outdoor-*, metres) skip the alignment (`output_kind: metric_depth`).

Everything downstream of the network (per-box statistic, anchors, alignment, fusion with the pinhole estimate) is
plain numpy, so `scripts/gate_replay.py` replays it on the CPU from the per-frame statistics stored at capture time,
like the rest of the geometry. The fusion itself lives in `MotionEstimator.update(..., depth=DepthFrame)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import CameraConfig, DepthConfig, resolve_model
from ..types import Detection
from .geometry import focal_length_px, ground_distance

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_DISTANCE_M = 200.0
MIN_DISTANCE_M = 1.0
# Fractions of the box (x1, x2, y1, y2) read as the object's body: skips the outline, where the map blends the
# object with the road and the background.
INNER_CROP = (0.25, 0.75, 0.35, 0.90)


def preprocess(frame_bgr: np.ndarray, width: int, height: int, normalize: bool = True) -> np.ndarray:
    """BGR frame -> [1, 3, height, width] float32 RGB in 0..1, ImageNet-normalised when `normalize`."""
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    if normalize:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class DepthModel:
    """Runs the network on one frame and returns its raw map (height, width) float32 at model resolution."""

    def __init__(self, cfg: DepthConfig):
        self.cfg = cfg
        self.width, self.height = int(cfg.input_size[0]), int(cfg.input_size[1])
        if cfg.backend == "transformers":
            import torch
            from transformers import AutoModelForDepthEstimation

            self._torch = torch
            self.dtype = getattr(torch, cfg.dtype)
            self.model = AutoModelForDepthEstimation.from_pretrained(resolve_model(cfg.model), dtype=self.dtype)
            self.model = self.model.to(cfg.device).eval()
            self.normalize = True
        elif cfg.backend == "onnx":
            import onnxruntime as ort

            self.session = ort.InferenceSession(cfg.onnx_path, providers=list(cfg.onnx_providers))
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            if len(inp.shape) == 4 and all(isinstance(v, int) for v in inp.shape[2:]):
                self.height, self.width = int(inp.shape[2]), int(inp.shape[3])  # static export wins
            self.normalize = cfg.onnx_normalize
        else:
            raise ValueError(f"Unknown depth backend: {cfg.backend}")

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = preprocess(frame_bgr, self.width, self.height, self.normalize)
        if self.cfg.backend == "transformers":
            torch = self._torch
            with torch.inference_mode():
                pixels = torch.from_numpy(x).to(self.model.device, dtype=self.dtype)
                dmap = self.model(pixel_values=pixels).predicted_depth.float().cpu().numpy()
        else:
            dmap = self.session.run(None, {self.input_name: x})[0]
        dmap = np.asarray(dmap, dtype=np.float32)
        while dmap.ndim > 2:
            dmap = dmap[0]
        return dmap


@dataclass
class DepthFrame:
    """What the rest of the pipeline needs from one depth map: one statistic per detection (same order as the
    detections it was built for) and road anchors (map value, 1 / metres). Small enough to store per frame."""

    kind: str  # relative_disparity | metric_depth
    stats: list[float | None]
    anchors: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "stats": list(self.stats), "anchors": [list(a) for a in self.anchors]}

    @classmethod
    def from_dict(cls, d: dict) -> DepthFrame:
        return cls(kind=d["kind"], stats=list(d["stats"]), anchors=[tuple(a) for a in d.get("anchors", [])])

    def subset(self, indices: list[int]) -> DepthFrame:
        """The frame restricted to some of its detections (e.g. after the ego-hood filter in a replay)."""
        return DepthFrame(self.kind, [self.stats[i] for i in indices], list(self.anchors))


def box_stat(dmap: np.ndarray, box: tuple[float, float, float, float], frame_w: int, frame_h: int,
             percentile: float = 50.0, higher_is_closer: bool = True) -> float | None:
    """Robust map value of one detection: a percentile (toward the camera) of the box's inner crop."""
    mh, mw = dmap.shape
    sx, sy = mw / frame_w, mh / frame_h
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    fx1, fx2, fy1, fy2 = INNER_CROP
    cx1, cx2 = int(math.floor((x1 + fx1 * bw) * sx)), int(math.ceil((x1 + fx2 * bw) * sx))
    cy1, cy2 = int(math.floor((y1 + fy1 * bh) * sy)), int(math.ceil((y1 + fy2 * bh) * sy))
    cx1, cy1, cx2, cy2 = max(0, cx1), max(0, cy1), min(mw, cx2), min(mh, cy2)
    if cx2 - cx1 < 1 or cy2 - cy1 < 1:  # tiny box at map resolution: whole box instead of the inner crop
        cx1, cy1 = max(0, int(math.floor(x1 * sx))), max(0, int(math.floor(y1 * sy)))
        cx2, cy2 = min(mw, max(cx1 + 1, int(math.ceil(x2 * sx)))), min(mh, max(cy1 + 1, int(math.ceil(y2 * sy))))
        if cx2 - cx1 < 1 or cy2 - cy1 < 1:
            return None
    vals = dmap[cy1:cy2, cx1:cx2]
    value = float(np.percentile(vals, percentile if higher_is_closer else 100.0 - percentile))
    return value if math.isfinite(value) else None


def ground_anchors(dmap: np.ndarray, frame_w: int, frame_h: int, boxes: list[tuple[float, float, float, float]],
                   focal_px: float, cam_height_m: float, rows: tuple[float, float] = (0.62, 0.88), n_rows: int = 8,
                   cols: tuple[float, float] = (0.40, 0.60), n_cols: int = 5, margin: float = 0.02,
                   ) -> list[tuple[float, float]]:
    """(map value, 1 / metres) of road points in front of the car: a grid of rows x columns in the lower-central
    image, skipping points inside (or within `margin` of) a detection box. The metric distance of a road row comes
    from the flat-road geometry already used for clipped boxes (`ground_distance`)."""
    mh, mw = dmap.shape
    sx, sy = mw / frame_w, mh / frame_h
    mx_px, my_px = margin * frame_w, margin * frame_h
    grown = [(x1 - mx_px, y1 - my_px, x2 + mx_px, y2 + my_px) for x1, y1, x2, y2 in boxes]
    pairs = []
    for ry in np.linspace(rows[0], rows[1], n_rows):
        y = float(ry) * frame_h
        dist = ground_distance(y, focal_px, frame_h, cam_height_m)
        if dist is None or dist > MAX_DISTANCE_M:
            continue
        for rx in np.linspace(cols[0], cols[1], n_cols):
            x = float(rx) * frame_w
            if any(bx1 <= x <= bx2 and by1 <= y <= by2 for bx1, by1, bx2, by2 in grown):
                continue
            mx, my = min(mw - 1, int(x * sx)), min(mh - 1, int(y * sy))
            patch = dmap[max(0, my - 1):my + 2, max(0, mx - 1):mx + 2]
            if patch.size:
                pairs.append((float(np.median(patch)), 1.0 / dist))
    return pairs


@dataclass
class Alignment:
    """1 / d = scale * value + shift, fitted on `n` inlier anchors with a relative residual `rel_rmse`."""

    scale: float
    shift: float
    n: int
    rel_rmse: float

    def distance(self, value: float | None) -> float | None:
        if value is None:
            return None
        inv = self.scale * value + self.shift
        if not math.isfinite(inv) or inv <= 1.0 / MAX_DISTANCE_M:
            return None
        return max(MIN_DISTANCE_M, 1.0 / inv)


def fit_scale_shift(pairs: list[tuple[float, float]], min_n: int = 6, max_rel_rmse: float = 0.25,
                    ) -> Alignment | None:
    """Robust least squares of 1/d = scale * value + shift (iteratively re-weighted, Huber weights).

    None when there are too few anchors, no spread in the values, a non-positive scale (the map does not get
    larger for closer points) or a relative residual above `max_rel_rmse`: the frame then keeps pinhole distances.
    """
    if len(pairs) < max(3, min_n):
        return None
    v = np.array([p[0] for p in pairs], dtype=np.float64)
    y = np.array([p[1] for p in pairs], dtype=np.float64)
    if v.std() <= 1e-6 * max(1.0, abs(float(v.mean()))):
        return None
    design = np.stack([v, np.ones_like(v)], axis=1)
    w = np.ones_like(v)
    floor = 0.02 * float(np.median(np.abs(y)))  # residuals within 2% of a typical 1/d are never outliers
    r, spread = y, 1.0
    for _ in range(4):  # iteratively re-weighted least squares (Huber weights on a robust residual scale)
        sol, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        r = y - (float(sol[0]) * v + float(sol[1]))
        spread = max(1.4826 * float(np.median(np.abs(r))), floor)
        w = np.minimum(1.0, 2.0 * spread / np.maximum(np.abs(r), 1e-12))
    inlier = np.abs(r) <= 3.0 * spread
    n = int(inlier.sum())
    if n < min_n:
        return None
    sol, *_ = np.linalg.lstsq(design[inlier], y[inlier], rcond=None)  # final plain fit on the inliers only
    scale, shift = float(sol[0]), float(sol[1])
    if scale <= 0:
        return None
    r = y[inlier] - (scale * v[inlier] + shift)
    rel = math.sqrt(float(np.mean(r ** 2))) / max(1e-9, float(np.mean(y[inlier])))
    if rel > max_rel_rmse:
        return None
    return Alignment(scale, shift, n, rel)


class DepthAligner:
    """Per-video scale/shift of a relative-depth model: re-fitted every frame, smoothed with a time constant, and
    kept for `align_max_age_s` when a frame has no reliable fit (a reset forgets it)."""

    def __init__(self, cfg: DepthConfig):
        self.cfg = cfg
        self.alignment: Alignment | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self.alignment, self._t = None, None

    def update(self, pairs: list[tuple[float, float]], t: float) -> Alignment | None:
        fit = fit_scale_shift(pairs, self.cfg.min_anchors, self.cfg.max_rel_rmse)
        if fit is not None:
            prev = self.alignment
            if prev is None or self._t is None or t < self._t or self.cfg.align_tau_s <= 0:
                self.alignment = fit
            else:
                a = 1.0 - math.exp(-(t - self._t) / self.cfg.align_tau_s)
                self.alignment = Alignment(a * fit.scale + (1 - a) * prev.scale, a * fit.shift + (1 - a) * prev.shift,
                                           fit.n, fit.rel_rmse)
            self._t = t
        elif self._t is not None and (t < self._t or t - self._t > self.cfg.align_max_age_s):
            self.reset()
        return self.alignment


def depth_frame(dmap: np.ndarray, frame_w: int, frame_h: int, detections: list[Detection], cfg: DepthConfig,
                camera: CameraConfig) -> DepthFrame:
    """Per-detection statistics + road anchors of one map (anchors are always computed for a relative model, the
    config decides at fusion time whether they are used, so a replay can switch them on and off)."""
    relative = cfg.output_kind != "metric_depth"
    stats = [box_stat(dmap, d.box, frame_w, frame_h, cfg.box_percentile, higher_is_closer=relative)
             for d in detections]
    anchors: list[tuple[float, float]] = []
    if relative:
        anchors = ground_anchors(dmap, frame_w, frame_h, [d.box for d in detections],
                                 focal_length_px(frame_w, camera.hfov_deg), camera.mount_height_m,
                                 rows=(float(cfg.ground_rows[0]), float(cfg.ground_rows[1])))
    return DepthFrame(kind=cfg.output_kind, stats=stats, anchors=anchors)


class DepthEstimator:
    """Network + per-frame statistics: what `MotionEstimator.update(..., depth=...)` consumes."""

    def __init__(self, cfg: DepthConfig, camera: CameraConfig):
        self.cfg, self.camera = cfg, camera
        self.model = DepthModel(cfg)

    def __call__(self, frame_bgr: np.ndarray, detections: list[Detection]) -> DepthFrame:
        h, w = frame_bgr.shape[:2]
        return depth_frame(self.model(frame_bgr), w, h, detections, self.cfg, self.camera)
