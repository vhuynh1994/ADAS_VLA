"""Frame sources: video files, camera index, RTSP/HTTP streams, image files and image folders."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FrameHistory:
    """Short ring buffer of recent frames, to fetch the frame `gap_s` before the current one.

    Used for the 2-frame VLM input (VLMConfig.prev_frame_s) by the pipeline and the dataset builders alike.
    With gap_s <= 0 it stores nothing.
    """

    def __init__(self, gap_s: float, slack_s: float = 0.5):
        self.gap_s = gap_s
        self.slack_s = slack_s
        self._frames: deque[tuple[float, np.ndarray]] = deque()

    def push(self, t: float, frame: np.ndarray) -> None:
        if self.gap_s <= 0:
            return
        self._frames.append((t, frame))
        while self._frames and t - self._frames[0][0] > self.gap_s + self.slack_s:
            self._frames.popleft()

    def before(self, t: float) -> np.ndarray | None:
        """Newest stored frame at least `gap_s` older than `t`; None until enough history exists."""
        best = None
        for ts, frame in self._frames:
            if ts > t - self.gap_s + 1e-6:
                break
            best = frame
        return best

    def reset(self) -> None:
        self._frames.clear()



def iter_frames(source: str, max_frames: int | None = None, image_fps: float = 10.0
                ) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, timestamp_s, frame_bgr).

    Video files use their own timeline (idx / fps) so results are reproducible regardless of processing
    speed; live cameras use wall-clock time.
    """
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        yield from _iter_images(files, max_frames, image_fps)
        return
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        yield from _iter_images([path], max_frames, image_fps)
        return

    live = source.isdigit() or "://" in source
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t0 = time.monotonic()
    idx = 0
    try:
        while max_frames is None or idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.monotonic() - t0 if live else idx / fps
            yield idx, t, frame
            idx += 1
    finally:
        cap.release()


def _iter_images(files: list[Path], max_frames: int | None, fps: float):
    for idx, f in enumerate(files[:max_frames] if max_frames else files):
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        yield idx, idx / fps, frame


def source_fps(source: str, default: float = 30.0) -> float:
    if Path(source).is_file() and Path(source).suffix.lower() not in IMAGE_EXTS:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return fps
    return default
