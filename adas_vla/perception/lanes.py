"""Lane-line detection: lane mask (classic color/edge, or YOLOP CNN) -> ROI -> Hough -> ego-lane fit + smoothing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import LaneInfo


class LaneDetector:
    def __init__(self, roi_top: float = 0.6, smoothing: float = 0.3, max_missing: int = 10):
        self.roi_top = roi_top
        self.smoothing = smoothing
        self.max_missing = max_missing
        self._fits: dict[str, tuple[float, float] | None] = {"left": None, "right": None}
        self._missing = {"left": 0, "right": 0}

    def reset(self) -> None:
        self._fits = {"left": None, "right": None}
        self._missing = {"left": 0, "right": 0}

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, (0, 190, 0), (180, 255, 255))
        yellow = cv2.inRange(hls, (15, 80, 90), (35, 220, 255))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        color = cv2.dilate(cv2.bitwise_or(white, yellow), np.ones((3, 3), np.uint8))
        mask = cv2.bitwise_and(edges, color)

        top = int(self.roi_top * h)
        roi = np.array([[
            (int(0.05 * w), h), (int(0.44 * w), top), (int(0.56 * w), top), (int(0.95 * w), h),
        ]], dtype=np.int32)
        roi_mask = np.zeros_like(mask)
        cv2.fillPoly(roi_mask, roi, 255)
        return cv2.bitwise_and(mask, roi_mask)

    def _fit_side(self, segments: list[tuple[int, int, int, int]]) -> tuple[float, float] | None:
        """Least-squares fit of x = m*y + b, weighting points by segment length."""
        if not segments:
            return None
        ys, xs, ws = [], [], []
        for x1, y1, x2, y2 in segments:
            length = float(np.hypot(x2 - x1, y2 - y1))
            ys += [y1, y2]
            xs += [x1, x2]
            ws += [length, length]
        m, b = np.polyfit(np.array(ys, float), np.array(xs, float), 1, w=np.sqrt(ws))
        return float(m), float(b)

    def __call__(self, frame_bgr: np.ndarray) -> LaneInfo:
        h, w = frame_bgr.shape[:2]
        mask = self._mask(frame_bgr)
        lines = cv2.HoughLinesP(mask, rho=2, theta=np.pi / 180, threshold=25,
                                minLineLength=int(0.03 * h), maxLineGap=int(0.15 * h))
        sides: dict[str, list] = {"left": [], "right": []}
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):  # (N,1,4) or (N,4) depending on OpenCV
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.4:  # near-horizontal: not a lane line
                    continue
                cx = (x1 + x2) / 2
                if slope < 0 and cx < 0.55 * w:
                    sides["left"].append((x1, y1, x2, y2))
                elif slope > 0 and cx > 0.45 * w:
                    sides["right"].append((x1, y1, x2, y2))

        y_top, y_bottom = self.roi_top * h, float(h)
        for side, segs in sides.items():
            fit = self._fit_side(self._nearest_line(side, segs, w, float(h)))
            if fit is not None and not self._plausible(side, fit, w, y_top, y_bottom):
                fit = None
            prev = self._fits[side]
            if fit is None:
                self._missing[side] += 1
                if self._missing[side] > self.max_missing:
                    self._fits[side] = None
                continue
            self._missing[side] = 0
            if prev is None:
                self._fits[side] = fit
            else:
                a = self.smoothing
                self._fits[side] = (a * fit[0] + (1 - a) * prev[0], a * fit[1] + (1 - a) * prev[1])

        lanes = LaneInfo(self._fits["left"], self._fits["right"], y_top, y_bottom, w)
        if lanes.valid:
            left, right = lanes.bounds_at(y_bottom)
            if not (0.25 * w < right - left < 1.5 * w):
                lanes = LaneInfo(None, None, y_top, y_bottom, w)
        return lanes

    @staticmethod
    def _nearest_line(side: str, segs: list, w: int, y_bottom: float) -> list:
        """Keep only segments of the line closest to the ego vehicle (drop adjacent-lane markings)."""
        if len(segs) < 2:
            return segs

        def x_bottom(seg):
            x1, y1, x2, y2 = seg
            return x1 + (x2 - x1) * (y_bottom - y1) / (y2 - y1) if y2 != y1 else x1

        xs = [x_bottom(sg) for sg in segs]
        center = w / 2
        candidates = [x for x in xs if (x < center if side == "left" else x > center)] or xs
        nearest = max(candidates) if side == "left" else min(candidates)
        return [sg for sg, x in zip(segs, xs) if abs(x - nearest) < 0.08 * w]

    @staticmethod
    def _plausible(side: str, fit: tuple[float, float], w: int, y_top: float, y_bottom: float) -> bool:
        x_bottom = LaneInfo.x_at(fit, y_bottom)
        x_top = LaneInfo.x_at(fit, y_top)
        if side == "left":
            return -0.5 * w < x_bottom < 0.55 * w and x_top < 0.6 * w
        return 0.45 * w < x_bottom < 1.5 * w and x_top > 0.4 * w


class YolopLaneDetector(LaneDetector):
    """Lane lines from the YOLOP lane-segmentation head (hustvl/YOLOP, MIT), run on the GPU.

    Only the two segmentation heads are kept (the detection head is dropped; YOLO11 handles objects),
    converted from ONNX to PyTorch so it runs in ~7 ms on an RTX 4060 in fp16.
    """

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, weights: str, device: str = "cuda:0", keep_drivable: bool = False, **kwargs):
        """keep_drivable: also publish the drivable-area mask as `self.drivable` (nothing in the pipeline reads it,
        so it is off by default to save an argmax + resize per frame)."""
        super().__init__(**kwargs)
        self.keep_drivable = keep_drivable

        import onnx
        import torch
        from onnx2torch import convert

        src = Path(weights)
        seg = src.with_name(src.stem.replace("yolop-", "yolop-seg-") + src.suffix)
        if not seg.exists():
            onnx.utils.extract_model(str(src), str(seg), ["images"], ["drive_area_seg", "lane_line_seg"])
        self.size = onnx.load(str(seg)).graph.input[0].type.tensor_type.shape.dim[2].dim_value
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        model = convert(onnx.load(str(seg))).eval().to(self.device)
        self.model = model.half() if self.half else model
        self.drivable: np.ndarray | None = None  # last drivable-area mask (H x W, uint8 0/1)

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch

        h, w = frame_bgr.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.MEAN) / self.STD
        x = torch.from_numpy(rgb.transpose(2, 0, 1)[None]).to(self.device)
        with torch.inference_mode():
            drive, lane = self.model(x.half() if self.half else x)
        lane = lane[0].argmax(0)[top:top + nh, left:left + nw].to(torch.uint8).cpu().numpy()
        if self.keep_drivable:
            drive = drive[0].argmax(0)[top:top + nh, left:left + nw].to(torch.uint8).cpu().numpy()
            self.drivable = cv2.resize(drive, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = cv2.resize(lane * 255, (w, h), interpolation=cv2.INTER_NEAREST)

        top_y = int(self.roi_top * h)
        roi = np.zeros_like(mask)
        roi[top_y:] = 255
        return cv2.bitwise_and(mask, roi)


def build_lane_detector(model: str, weights: str, device: str) -> LaneDetector:
    """YOLOP when its weights are available, otherwise the classic detector."""
    if model == "yolop":
        path = Path(weights)
        if not path.is_absolute():
            from ..config import PROJECT_ROOT

            path = PROJECT_ROOT / path
        if path.exists():
            return YolopLaneDetector(str(path), device)
        import logging

        logging.getLogger(__name__).warning("YOLOP weights not found at %s - using classic lane detection", path)
    return LaneDetector()
