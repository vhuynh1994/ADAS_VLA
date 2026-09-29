"""End-to-end ADAS pipeline: fast perception every frame, slow VLM reasoning every N frames,
deterministic safety arbitration and control on every frame (dual-rate "System 1 / System 2")."""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from .config import Config
from .control import Controller, SafetySupervisor
from .perception import MotionEstimator
from .perception.lanes import build_lane_detector
from .sources import FrameHistory
from .types import VRU_CLASSES, DrivingDecision, EgoState, FrameResult, LaneInfo, SceneContext

log = logging.getLogger(__name__)


class _AsyncVLMWorker:
    """Runs the VLM on the most recent submitted frame in a background thread (drops older frames)."""

    def __init__(self, run_fn):
        self._run_fn = run_fn
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._result: DrivingDecision | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vlm-worker")
        self._thread.start()

    def submit(self, frames: tuple[np.ndarray, np.ndarray | None], ctx: SceneContext) -> None:
        try:
            self._inbox.get_nowait()  # drop the stale pending frame
        except queue.Empty:
            pass
        self._inbox.put_nowait((tuple(None if f is None else f.copy() for f in frames), ctx))

    def latest(self) -> DrivingDecision | None:
        with self._lock:
            return self._result

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames, ctx = self._inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            decision = self._run_fn(frames, ctx)
            if decision is not None:
                with self._lock:
                    self._result = decision

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


class ADASPipeline:
    def __init__(self, cfg: Config, vlm=None, load_vlm: bool = True):
        from .perception.detector import ObjectDetector

        self.cfg = cfg
        self.detector = ObjectDetector(cfg.perception)
        self.lane_detector = build_lane_detector(
            cfg.perception.lane_model, cfg.perception.lane_weights, cfg.perception.device,
        ) if cfg.perception.lane_detection else None
        self.motion = MotionEstimator(cfg.camera, cut_in_rate=cfg.perception.cut_in_rate,
                                      cut_in_max_distance_m=cfg.perception.cut_in_max_distance_m,
                                      cut_in_max_pull_away_mps=cfg.perception.cut_in_max_pull_away_mps,
                                      vel_window_s=cfg.perception.velocity_window_s)
        self.safety = SafetySupervisor(cfg)
        self.controller = Controller(cfg.control)
        self._history = FrameHistory(cfg.vlm.prev_frame_s)  # earlier frame for the 2-frame VLM input

        self.vlm = vlm
        if self.vlm is None and load_vlm and cfg.vlm.enabled:
            from .reasoning.vlm import VisionLanguageModel

            log.info("Loading VLM %s (%s)...", cfg.vlm.model_id, cfg.vlm.quantization)
            self.vlm = VisionLanguageModel(cfg.vlm)

        self.latest_vlm: DrivingDecision | None = None
        self.last_raw: str = ""
        self.vlm_calls = 0
        self.vlm_failures = 0
        self.vlm_latencies: list[float] = []
        self._worker = _AsyncVLMWorker(self._run_vlm) if self.vlm and cfg.vlm.mode == "async" else None
        self._last_vlm_frame: int | None = None
        self._last_signature: tuple | None = None

    def perceive(self, frame: np.ndarray, frame_idx: int, t: float, ego: EgoState) -> SceneContext:
        h, w = frame.shape[:2]
        detections = self.detector(frame)
        lanes = self.lane_detector(frame) if self.lane_detector else LaneInfo(image_width=w)
        self.motion.update(detections, t, lanes, w, h, ego.speed_mps)
        return SceneContext(frame_idx=frame_idx, timestamp_s=t, width=w, height=h, ego=ego,
                            detections=detections, lanes=lanes)

    @staticmethod
    def scene_signature(ctx: SceneContext) -> tuple:
        """Coarse description of what matters for a decision; a change triggers a VLM call."""
        lead = ctx.lead_object()
        gap = None if lead is None or lead.distance_m is None else min(3, int(lead.distance_m // 15))
        vru = any(d.cls_name in VRU_CLASSES and d.in_ego_path for d in ctx.detections)
        red = any(d.cls_name == "traffic light" and d.attribute == "red" for d in ctx.detections)
        cut_in = any(d.cutting_in for d in ctx.detections)
        return (lead.track_id if lead else None, gap, vru, red, cut_in, ctx.lanes.valid)

    def should_call_vlm(self, ctx: SceneContext) -> bool:
        v = self.cfg.vlm
        signature = self.scene_signature(ctx)
        changed = signature != self._last_signature
        self._last_signature = signature
        if self._last_vlm_frame is None:
            return True
        since = ctx.frame_idx - self._last_vlm_frame
        if since >= max(1, v.every_n_frames):
            return True
        return v.trigger == "event" and changed and since >= v.min_interval_frames

    def _run_vlm(self, frames: tuple[np.ndarray, np.ndarray | None], ctx: SceneContext) -> DrivingDecision | None:
        frame, prev = frames
        decision, raw, latency = self.vlm.decide(
            frame, ctx.summary_text(), ctx.ego.speed_kmh, self.cfg.control.cruise_speed_kmh,
            self.cfg.safety.max_speed_kmh, prev_image=prev,
        )
        self.vlm_calls += 1
        self.vlm_latencies.append(latency)
        self.last_raw = raw
        if decision is None:
            self.vlm_failures += 1
            log.warning("Unparsable VLM output (frame %d): %s", ctx.frame_idx, raw[:200])
            return None
        decision.frame_idx = ctx.frame_idx
        decision.timestamp_s = ctx.timestamp_s
        return decision

    def process(self, frame: np.ndarray, frame_idx: int, t: float, ego: EgoState) -> FrameResult:
        t0 = time.perf_counter()
        ctx = self.perceive(frame, frame_idx, t, ego)
        perception_ms = (time.perf_counter() - t0) * 1000

        self._history.push(t, frame)
        if self.vlm is not None and self.should_call_vlm(ctx):
            self._last_vlm_frame = frame_idx
            frames = (frame, self._history.before(t))
            if self._worker is not None:
                self._worker.submit(frames, ctx)
            else:
                decision = self._run_vlm(frames, ctx)
                if decision is not None:
                    self.latest_vlm = decision
        if self._worker is not None:
            self.latest_vlm = self._worker.latest() or self.latest_vlm

        final, alerts = self.safety.arbitrate(ctx, self.latest_vlm)
        command = self.controller(final, ctx)
        return FrameResult(context=ctx, decision=final, vlm_decision=self.latest_vlm, command=command,
                           alerts=alerts, perception_ms=perception_ms)

    def reset(self) -> None:
        self.detector.reset()
        self.motion.reset()
        self.safety.reset()
        self._history.reset()
        if self.lane_detector:
            self.lane_detector.reset()
        self.latest_vlm = None
        self._last_vlm_frame = None
        self._last_signature = None

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
