"""Heads-up display overlay. Shapes are drawn with OpenCV; text with PIL so Vietnamese renders correctly."""

from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import FrameResult, LaneInfo, LongAction

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
ACTION_COLORS = {  # BGR
    LongAction.EMERGENCY_BRAKE: (0, 0, 255), LongAction.BRAKE: (0, 60, 255), LongAction.STOP: (0, 60, 255),
    LongAction.DECELERATE: (0, 165, 255), LongAction.ACCELERATE: (80, 200, 80),
}


def _font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class HUD:
    def __init__(self):
        self._fonts = {s: _font(s) for s in (14, 16, 20, 28)}

    def draw(self, frame: np.ndarray, res: FrameResult, copilot: str | None = None) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        texts: list[tuple[tuple[int, int], str, int, tuple[int, int, int]]] = []  # (xy, text, size, BGR)

        self._draw_lanes(img, res.context.lanes, h)

        for d in res.context.detections:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            critical = d.threatening and d.ttc_s is not None and d.ttc_s < 2.7
            color = (0, 0, 255) if critical else (0, 165, 255) if d.threatening else (80, 200, 80)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = d.cls_name if not d.attribute else f"{d.cls_name}:{d.attribute}"
            if d.distance_m is not None:
                label += f" {d.distance_m:.0f}m"
            if d.ttc_s is not None and d.in_ego_path:
                label += f" ttc{d.ttc_s:.1f}"
            if d.oncoming:
                label += " oncoming"
            if d.cutting_in:
                label += " cut-in"

            texts.append(((x1, max(0, y1 - 18)), label, 14, color))

        # Status panel (top-left).
        panel_w, panel_h = 400, 150
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        dec, cmd = res.decision, res.command
        texts.append(((10, 8), dec.label, 20, ACTION_COLORS.get(dec.longitudinal, (255, 255, 255))))
        texts.append(((10, 42), f"ego {res.context.ego.speed_kmh:.0f} km/h -> target {dec.target_speed_kmh:.0f} km/h",
                      16, (255, 255, 255)))
        texts.append(((10, 64), f"source: {dec.source}   risk: {dec.risk_level.value}", 16, (200, 200, 200)))
        self._bar(img, 10, 92, cmd.throttle, (80, 200, 80), "throttle", texts)
        self._bar(img, 10, 112, cmd.brake, (0, 0, 255), "brake", texts)
        cx = 300
        cv2.line(img, (cx, 100), (cx + int(cmd.steer * 60), 100), (255, 200, 0), 4)
        cv2.circle(img, (cx, 100), 4, (255, 255, 255), -1)
        texts.append(((cx - 25, 110), f"steer {cmd.steer:+.2f}", 14, (255, 200, 0)))
        texts.append(((10, 130), f"perception {res.perception_ms:.0f} ms", 14, (160, 160, 160)))

        # Alerts banner (top-right).
        y = 8
        for a in res.alerts:
            color = (0, 0, 255) if a.level == "critical" else (0, 200, 255) if a.level == "warning" else (255, 200, 0)
            texts.append(((w - 460, y), f"[{a.kind}] {a.message}", 20, color))
            y += 28

        # VLM reasoning and LLM copilot explanation (bottom).
        vlm = res.vlm_decision
        lines = []
        if copilot:
            lines.append("LLM copilot: " + copilot)
        if vlm is not None:
            age = res.context.timestamp_s - vlm.timestamp_s
            lines.append(f"VLM ({vlm.latency_s:.1f}s, {age:.1f}s ago): {vlm.label}, {vlm.target_speed_kmh:.0f} km/h")
            if vlm.reason:
                lines.append("reason: " + vlm.reason)
        if lines:
            wrapped = [seg for line in lines for seg in textwrap.wrap(line, width=max(40, w // 9))][:6]
            box_h = 10 + 22 * len(wrapped)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - box_h), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
            for i, line in enumerate(wrapped):
                texts.append(((10, h - box_h + 5 + 22 * i), line, 16, (255, 255, 255)))

        return self._render_text(img, texts)

    @staticmethod
    def _draw_lanes(img: np.ndarray, lanes: LaneInfo, h: int) -> None:
        if lanes.valid:
            yt, yb = int(lanes.y_top), h
            pts = np.array([
                (int(lanes.x_at(lanes.left_fit, yb)), yb), (int(lanes.x_at(lanes.left_fit, yt)), yt),
                (int(lanes.x_at(lanes.right_fit, yt)), yt), (int(lanes.x_at(lanes.right_fit, yb)), yb),
            ], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        for fit in (lanes.left_fit, lanes.right_fit):
            if fit is not None:
                p1 = (int(lanes.x_at(fit, lanes.y_top)), int(lanes.y_top))
                p2 = (int(lanes.x_at(fit, h)), h)
                cv2.line(img, p1, p2, (0, 255, 255), 3)

    @staticmethod
    def _bar(img, x, y, value, color, label, texts):
        cv2.rectangle(img, (x + 70, y), (x + 170, y + 12), (90, 90, 90), 1)
        cv2.rectangle(img, (x + 70, y), (x + 70 + int(100 * value), y + 12), color, -1)
        texts.append(((x, y - 3), label, 14, (220, 220, 220)))

    def _render_text(self, img: np.ndarray, texts) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for (x, y), text, size, (b, g, r) in texts:
            draw.text((x, y), text, font=self._fonts.get(size, self._fonts[16]), fill=(r, g, b),
                      stroke_width=1, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
