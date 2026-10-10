"""Object detection + multi-object tracking with Ultralytics YOLO."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PerceptionConfig
from ..types import Detection


def traffic_light_color(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> str | None:
    """Classify a traffic light crop as red/yellow/green by counting bright saturated pixels."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 1] > 90) & (hsv[..., 2] > 150)
    hue = hsv[..., 0]
    counts = {
        "red": int(np.count_nonzero(bright & ((hue < 10) | (hue > 165)))),
        "yellow": int(np.count_nonzero(bright & (hue >= 15) & (hue <= 35))),
        "green": int(np.count_nonzero(bright & (hue >= 45) & (hue <= 95))),
    }
    color, n = max(counts.items(), key=lambda kv: kv[1])
    return color if n >= max(4, 0.03 * bright.size) else None


def is_ego_hood(box: tuple[float, float, float, float], width: int, height: int) -> bool:
    """The ego car's own hood/dashboard is often detected as a 'car' spanning the bottom of the frame.

    A box reaching the bottom that spans nearly the whole width is the ego car (hood + dashboard + windscreen
    frame) whatever its top: a real vehicle fills 90% of a 60 deg view only within ~2 m, where it was already
    tracked while approaching (AEB acts earlier), so dropping such boxes costs no safety.
    """
    x1, y1, x2, y2 = box
    if y2 <= 0.9 * height:
        return False
    return (x2 - x1) > 0.9 * width or ((x2 - x1) > 0.55 * width and y1 > 0.5 * height)


class ObjectDetector:
    def __init__(self, cfg: PerceptionConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.detector_model)
        names = self.model.names
        wanted = set(cfg.classes)
        self.class_ids = [i for i, n in names.items() if n in wanted]
        self.names = names
        self.last_timing: dict[str, float] = {}  # Ultralytics split of the last call, ms (det_pre/model/post)

    def reset(self) -> None:
        """Forget tracker state (call when switching to a new video)."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for tracker in predictor.trackers:
                tracker.reset()

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        kwargs = dict(
            conf=self.cfg.conf, imgsz=self.cfg.imgsz, classes=self.class_ids,
            device=self.cfg.device, verbose=False,
        )
        if self.cfg.track:
            result = self.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
        else:
            result = self.model.predict(frame_bgr, **kwargs)[0]
        speed = result.speed or {}
        self.last_timing = {name: float(speed[key]) for name, key in
                            (("det_pre", "preprocess"), ("det_model", "inference"), ("det_post", "postprocess"))
                            if speed.get(key) is not None}

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

        h, w = frame_bgr.shape[:2]
        detections = []
        for box, conf, cls, tid in zip(xyxy, confs, clss, ids):
            if is_ego_hood(box, w, h):
                continue
            name = self.names[int(cls)]
            det = Detection(
                cls_name=name, conf=float(conf), box=tuple(float(v) for v in box),
                track_id=None if tid is None else int(tid),
            )
            if name == "traffic light":
                det.attribute = traffic_light_color(frame_bgr, det.box)
            detections.append(det)
        return detections
