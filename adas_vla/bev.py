"""Bird's-eye view (BEV) of one frame: lane lines and detected objects on the road plane, seen from above.

Same flat-road pinhole model as perception/geometry.py: focal f = (W/2) / tan(hfov/2); a road point seen at image
row y lies Z = f * h / (y - horizon) metres ahead and X = (x - W/2) * h / (y - horizon) metres to the side
(h = camera height, horizon = the row the geometry used for this frame: the lane vanishing point smoothed over
time, `SceneContext.horizon_y`). Objects use their estimated distance (`Detection.distance_m`, box height) and
X = (cx - W/2) * Z / f. Everything is a monocular estimate: with h = 1.3 m, a 0.5 degree pitch error moves a road
point seen at 40 m to 32-55 m, so far lane points are drawn only up to `lane_range_m`.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import CameraConfig, Config
from .hud import _font
from .perception.geometry import REAL_WIDTH_M, focal_length_px
from .types import Detection, SceneContext

LENGTH_M = {"person": 0.5, "bicycle": 1.8, "car": 4.5, "motorcycle": 2.0, "bus": 12.0, "truck": 8.0}
EGO_LENGTH_M, EGO_WIDTH_M = 4.5, 1.8
MIN_DY_FRAC = 0.03  # lane points closer than this to the horizon (fraction of the height) are not projected
# BGR, matching the HUD: critical red, threatening orange, normal green
BG, GRID, TEXT_MUTED = (34, 30, 28), (70, 64, 60), (175, 170, 165)
CRITICAL, THREAT, NORMAL, ONCOMING, LANE = (0, 0, 255), (0, 165, 255), (80, 200, 80), (190, 150, 110), (0, 230, 255)
LIGHT_COLORS = {"red": (0, 0, 255), "yellow": (0, 200, 255), "green": (80, 200, 80)}


def road_point(x: float, y: float, width: int, height: int, horizon_y: float, cam_h: float,
               focal_px: float) -> tuple[float, float] | None:
    """(X lateral, + = right; Z ahead) in metres of the road point seen at image (x, y), or None near/above the
    horizon."""
    dy = y - horizon_y
    if dy < MIN_DY_FRAC * height:
        return None
    return (x - width / 2) * cam_h / dy, focal_px * cam_h / dy


def object_xz(det: Detection, width: int, focal_px: float) -> tuple[float, float] | None:
    """(X, Z) of the visible face of a detected object from its estimated distance and its box centre."""
    if det.distance_m is None:
        return None
    z = det.distance_m
    return ((det.box[0] + det.box[2]) / 2 - width / 2) * z / focal_px, z


class BEVRenderer:
    """Draws the BEV panel; `render(ctx, height)` returns a BGR image `width` x `height` to put beside the HUD."""

    def __init__(self, camera: CameraConfig, corridor_half_width_m: float | None = 1.2, range_m: float = 60.0,
                 lane_range_m: float = 45.0, width: int = 360):
        self.camera = camera
        self.corridor_half_width_m = corridor_half_width_m
        self.range_m = range_m
        self.lane_range_m = lane_range_m
        self.width = width
        self._fonts = {s: _font(s) for s in (12, 14, 16)}

    @classmethod
    def from_config(cls, cfg: Config) -> BEVRenderer:
        return cls(cfg.camera, cfg.perception.corridor_half_width_m)

    # ------------------------------------------------------------------ layout
    def _scale(self, height: int) -> tuple[float, int, int]:
        top, bottom = 36, 30
        return (height - top - bottom) / (self.range_m + EGO_LENGTH_M), top, bottom

    def _px(self, x_m: float, z_m: float, height: int) -> tuple[int, int]:
        s, _, bottom = self._scale(height)
        return (round(self.width / 2 + x_m * s), round(height - bottom - (z_m + EGO_LENGTH_M) * s))

    # ------------------------------------------------------------------ drawing
    def render(self, ctx: SceneContext, height: int) -> np.ndarray:
        img = np.full((height, self.width, 3), BG, np.uint8)
        texts: list[tuple[tuple[int, int], str, int, tuple[int, int, int]]] = []
        s, top, bottom = self._scale(height)
        half_view = self.width / 2 / s

        for z in range(10, int(self.range_m) + 1, 10):  # range rings every 10 m
            _, yz = self._px(0, z, height)
            cv2.line(img, (0, yz), (self.width, yz), GRID, 1)
            texts.append(((4, yz - 15), f"{z} m", 12, TEXT_MUTED))
        for xm in range(5, int(half_view) + 1, 5):
            for sx in (-xm, xm):
                xp, _ = self._px(sx, 0, height)
                cv2.line(img, (xp, top), (xp, height - bottom), GRID, 1, cv2.LINE_AA)

        if self.corridor_half_width_m:  # metric ego corridor the gate uses for vehicles
            c = self.corridor_half_width_m
            overlay = img.copy()
            cv2.rectangle(overlay, self._px(-c, self.range_m, height), self._px(c, 0, height), (110, 90, 40), -1)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        f_px = focal_length_px(ctx.width, self.camera.hfov_deg)
        horizon = getattr(ctx, "horizon_y", None)
        if horizon is None:
            horizon = self.camera.horizon_frac * ctx.height
        self._draw_lanes(img, ctx, horizon, f_px, height)

        lead = ctx.lead_object()
        for det in sorted(ctx.detections, key=lambda d: -(d.distance_m or 0)):  # far first, near on top
            self._draw_object(img, det, det is lead, ctx.width, f_px, height, texts)

        x1, y1 = self._px(-EGO_WIDTH_M / 2, 0, height)
        x2, y2 = self._px(EGO_WIDTH_M / 2, -EGO_LENGTH_M, height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (235, 235, 235), -1)
        cv2.rectangle(img, (x1, y1), (x2, y1 + 6), (120, 120, 120), -1)  # windshield / camera side

        texts.append(((8, 8), "BEV · ước lượng 1 camera", 16, (235, 235, 235)))
        legend = f"lưới 10 m · hành lang ±{self.corridor_half_width_m:g} m" if self.corridor_half_width_m else "lưới 10 m"
        texts.append(((8, height - bottom + 7), legend, 12, TEXT_MUTED))
        return self._render_text(img, texts)

    def _draw_lanes(self, img: np.ndarray, ctx: SceneContext, horizon: float, f_px: float, height: int) -> None:
        lanes = ctx.lanes
        y_far = max(lanes.y_top, horizon + MIN_DY_FRAC * ctx.height)
        rows = np.linspace(ctx.height, y_far, 40)
        polys = []
        for fit in (lanes.left_fit, lanes.right_fit):
            if fit is None:
                polys.append(None)
                continue
            pts = []
            for y in rows:
                p = road_point(lanes.x_at(fit, y), y, ctx.width, ctx.height, horizon, self.camera.mount_height_m, f_px)
                if p is not None and 0 <= p[1] <= self.lane_range_m:
                    pts.append(self._px(p[0], p[1], height))
            polys.append(np.array(pts, np.int32) if len(pts) >= 2 else None)
        if polys[0] is not None and polys[1] is not None:  # ego lane area, as on the HUD
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.vstack([polys[0], polys[1][::-1]])], (0, 150, 0))
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        for poly in polys:
            if poly is not None:
                cv2.polylines(img, [poly], False, LANE, 2, cv2.LINE_AA)

    def _draw_object(self, img, det: Detection, is_lead: bool, frame_w: int, f_px: float, height: int, texts) -> None:
        p = object_xz(det, frame_w, f_px)
        if p is None:
            return
        x, z = p
        if z > self.range_m:  # beyond the panel: a marker on the top edge, on the side it is
            if det.cls_name in ("traffic light", "stop sign"):
                return
            _, top, _ = self._scale(height)
            cx = min(max(self._px(x * self.range_m / z, self.range_m, height)[0], 6), self.width - 6)
            color = ONCOMING if det.oncoming else NORMAL
            cv2.fillPoly(img, [np.array([(cx, top + 2), (cx - 5, top + 11), (cx + 5, top + 11)], np.int32)], color)
            return
        if det.cls_name in ("traffic light", "stop sign"):
            color = LIGHT_COLORS.get(det.attribute or "", (200, 200, 200)) if det.cls_name == "traffic light" else (0, 0, 220)
            cx, cy = self._px(x, z, height)
            cv2.circle(img, (cx, cy), 5, color, -1, cv2.LINE_AA)
            return
        critical = det.threatening and det.ttc_s is not None and det.ttc_s < 2.7
        color = (CRITICAL if critical else THREAT if det.threatening else ONCOMING if det.oncoming else NORMAL)
        w = REAL_WIDTH_M.get(det.cls_name, 0.8)
        length = LENGTH_M.get(det.cls_name, 1.0)
        x1, y1 = self._px(x - w / 2, z + length, height)
        x2, y2 = self._px(x + w / 2, z, height)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1 if det.threatening else 2)
        if is_lead:
            cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 1)
        label = f"{det.cls_name} {z:.0f}m"
        if det.ttc_s is not None and det.in_ego_path:
            label += f" ttc{det.ttc_s:.1f}"
        if det.cutting_in:
            label += " cut-in"
        lx = x2 + 4 if x2 + 4 + 7 * len(label) < self.width else max(0, x1 - 4 - 7 * len(label))
        texts.append(((lx, y2 - 14), label, 12, color))

    def _render_text(self, img: np.ndarray, texts) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for (x, y), text, size, (b, g, r) in texts:
            draw.text((x, y), text, font=self._fonts.get(size, self._fonts[14]), fill=(r, g, b),
                      stroke_width=1, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def pitch_error_range(distance_m: float, cam_h: float, pitch_err_deg: float) -> tuple[float, float]:
    """Road-point distances a +-pitch error turns `distance_m` into (used to document the BEV's uncertainty)."""
    a = math.atan2(cam_h, distance_m)
    e = math.radians(pitch_err_deg)
    near = cam_h / math.tan(a + e)
    far = cam_h / math.tan(a - e) if a > e else math.inf
    return near, far
