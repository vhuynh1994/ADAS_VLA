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
from .timing import StageTimer
from .types import VRU_CLASSES, DrivingDecision, EgoState, FrameResult, LaneInfo, SceneContext

log = logging.getLogger(__name__)


class _AsyncVLMWorker:
    """Runs the VLM on the most recent submitted frame in a background thread (drops older frames)."""

    def __init__(self, run_fn):
        self._run_fn = run_fn
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._result: tuple[DrivingDecision, float] | None = None
        self._generation = 0  # bumped by clear(): results of calls submitted before it are dropped
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vlm-worker")
        self._thread.start()

    def submit(self, frames: tuple[np.ndarray, np.ndarray | None], ctx: SceneContext, capture_s: float) -> None:
        try:
            self._inbox.get_nowait()  # drop the stale pending frame
        except queue.Empty:
            pass
        self._inbox.put_nowait((tuple(None if f is None else f.copy() for f in frames), ctx, capture_s,
                                self._generation))

    def latest(self) -> tuple[DrivingDecision, float] | None:
        """(decision, capture time of its frame) of the newest successful call."""
        with self._lock:
            return self._result

    def clear(self) -> None:
        """Forget the latest decision and any pending frame (new video: its timeline restarts at 0)."""
        try:
            self._inbox.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._generation += 1
            self._result = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames, ctx, capture_s, generation = self._inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            decision = self._run_fn(frames, ctx)
            if decision is not None:
                with self._lock:
                    if generation == self._generation:
                        self._result = (decision, capture_s)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _vlm_process_main(make_vlm, cruise_kmh: float, max_kmh: float, inbox, outbox) -> None:
    """Child process of _ProcessVLMWorker: load the VLM, then answer one request at a time."""
    import signal
    import traceback

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C goes to the parent, which shuts us down
    try:
        vlm = make_vlm()
    except Exception:
        outbox.put(("error", traceback.format_exc()))
        return
    outbox.put(("ready", None))
    while (request := inbox.get()) is not None:
        generation, frame, prev, text, ego_kmh = request
        try:
            decision, raw, latency = vlm.decide(frame, text, ego_kmh, cruise_kmh, max_kmh, prev_image=prev)
        except Exception:
            outbox.put(("error", traceback.format_exc()))
            return
        outbox.put(("result", (generation, decision, raw, latency)))


class _ProcessVLMWorker:
    """Same contract as _AsyncVLMWorker, with the VLM in a separate process.

    A VLM thread in the perception process holds the GIL while it waits for the GPU (measured on an RTX 4060:
    each prefill stalled the detector for ~240 ms); a process does not. The newest submitted frame is sent when
    the child is idle (frames submitted while it is busy replace each other), polled from the frame loop.
    """

    def __init__(self, make_vlm, cruise_kmh: float, max_kmh: float, on_result, start_timeout_s: float = 600.0):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # CUDA cannot be re-initialized in a forked child
        self._inbox, self._outbox = ctx.Queue(), ctx.Queue()
        self._on_result = on_result  # (decision | None, raw, latency, ctx) -> decision | None, in this process
        self._proc = ctx.Process(target=_vlm_process_main, args=(make_vlm, cruise_kmh, max_kmh, self._inbox,
                                                                 self._outbox), daemon=True, name="vlm-process")
        self._proc.start()
        self._pending = None  # newest submission not sent yet
        self._in_flight = None  # (generation, ctx, capture_s) of the request being processed
        self._generation = 0
        self._result: tuple[DrivingDecision, float] | None = None
        kind, payload = self._outbox.get(timeout=start_timeout_s)
        if kind != "ready":
            self._proc.join(timeout=5)
            raise RuntimeError(f"VLM process failed to start:\n{payload}")

    def submit(self, frames: tuple[np.ndarray, np.ndarray | None], ctx: SceneContext, capture_s: float) -> None:
        self._pending = (frames, ctx, capture_s)
        self._poll()

    def latest(self) -> tuple[DrivingDecision, float] | None:
        self._poll()
        return self._result

    def _poll(self) -> None:
        while self._in_flight is not None:
            try:
                kind, payload = self._outbox.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                raise RuntimeError(f"VLM process failed:\n{payload}")
            generation, decision, raw, latency = payload
            sent_generation, ctx, capture_s = self._in_flight
            self._in_flight = None
            if generation == self._generation == sent_generation:
                decision = self._on_result(decision, raw, latency, ctx)
                if decision is not None:
                    self._result = (decision, capture_s)
        if self._in_flight is None and self._pending is not None:
            (frame, prev), ctx, capture_s = self._pending
            self._pending = None
            self._inbox.put((self._generation, frame, prev, ctx.summary_text(), ctx.ego.speed_kmh))
            self._in_flight = (self._generation, ctx, capture_s)

    def clear(self) -> None:
        self._generation += 1
        self._pending = None
        self._result = None

    def close(self) -> None:
        if self._proc.is_alive():
            self._inbox.put(None)
            self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()


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
        in_process = cfg.vlm.mode == "process" and self.vlm is None and load_vlm and cfg.vlm.enabled
        if self.vlm is None and load_vlm and cfg.vlm.enabled and not in_process:
            from .reasoning.vlm import VisionLanguageModel

            log.info("Loading VLM %s (%s)...", cfg.vlm.model_id, cfg.vlm.quantization)
            self.vlm = VisionLanguageModel(cfg.vlm)

        self.latest_vlm: DrivingDecision | None = None
        self._latest_vlm_capture_s: float | None = None  # perf_counter capture time of latest_vlm's frame
        self.last_raw: str = ""
        self.vlm_calls = 0
        self.vlm_failures = 0
        self.vlm_latencies: list[float] = []
        self._worker = None
        if in_process:
            import functools

            from .reasoning.vlm import VisionLanguageModel

            log.info("Loading VLM %s (%s) in a separate process...", cfg.vlm.model_id, cfg.vlm.quantization)
            self._worker = _ProcessVLMWorker(functools.partial(VisionLanguageModel, cfg.vlm),
                                             cfg.control.cruise_speed_kmh, cfg.safety.max_speed_kmh, self._record_vlm)
        elif self.vlm is not None and cfg.vlm.mode in ("async", "process"):  # given model object: thread
            self._worker = _AsyncVLMWorker(self._run_vlm)
        self._vlm_active = self.vlm is not None or self._worker is not None
        self._last_vlm_frame: int | None = None
        self._last_signature: tuple | None = None

    def perceive(self, frame: np.ndarray, frame_idx: int, t: float, ego: EgoState,
                 timer: StageTimer | None = None) -> SceneContext:
        timer = timer or StageTimer()
        h, w = frame.shape[:2]
        with timer("detector"):
            detections = self.detector(frame)
        split = getattr(self.detector, "last_timing", {})
        if split:  # the rest of the call: ByteTrack + box conversion
            timer.update({**split, "det_track": timer.ms["detector"] - sum(split.values())})
        if self.lane_detector:
            with timer("lanes"):
                lanes = self.lane_detector(frame)
            split = self.lane_detector.last_timing
            if split:  # the rest of the call: mask post-processing + line fit
                timer.update({**split, "lane_post": timer.ms["lanes"] - sum(split.values())})
        else:
            lanes = LaneInfo(image_width=w)
        with timer("geometry"):
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
        return self._record_vlm(decision, raw, latency, ctx)

    def _record_vlm(self, decision: DrivingDecision | None, raw: str, latency: float,
                    ctx: SceneContext) -> DrivingDecision | None:
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

    def process(self, frame: np.ndarray, frame_idx: int, t: float, ego: EgoState,
                capture_s: float | None = None, read_ms: float | None = None) -> FrameResult:
        """capture_s: time.perf_counter() when the frame became available (default: now), the start of the
        capture -> command latency; read_ms: time spent reading / decoding it, logged as the `read` stage."""
        t0 = time.perf_counter()
        capture_s = t0 if capture_s is None else capture_s
        timer = StageTimer()
        if read_ms is not None:
            timer.add("read", read_ms)
        ctx = self.perceive(frame, frame_idx, t, ego, timer)
        perception_ms = (time.perf_counter() - t0) * 1000

        self._history.push(t, frame)
        with timer("vlm"):
            if self._vlm_active and self.should_call_vlm(ctx):
                self._last_vlm_frame = frame_idx
                frames = (frame, self._history.before(t))
                if self._worker is not None:
                    self._worker.submit(frames, ctx, capture_s)
                else:
                    decision = self._run_vlm(frames, ctx)
                    if decision is not None:
                        self.latest_vlm, self._latest_vlm_capture_s = decision, capture_s
            if self._worker is not None:
                latest = self._worker.latest()
                if latest is not None:
                    self.latest_vlm, self._latest_vlm_capture_s = latest

        with timer("gate"):
            final, alerts = self.safety.arbitrate(ctx, self.latest_vlm)
        with timer("control"):
            command = self.controller(final, ctx)
        done = time.perf_counter()
        e2e = (done - capture_s) * 1000 + (read_ms or 0.0)
        timer.add("e2e", e2e)
        timer.add("frame", e2e - timer.ms.get("vlm", 0.0))
        age = age_wall = None
        if self.latest_vlm is not None:
            age = ctx.timestamp_s - self.latest_vlm.timestamp_s
            age_wall = done - self._latest_vlm_capture_s
        return FrameResult(context=ctx, decision=final, vlm_decision=self.latest_vlm, command=command,
                           alerts=alerts, perception_ms=perception_ms, timing=timer.ms,
                           vlm_age_s=age, vlm_age_wall_s=age_wall)

    def reset(self) -> None:
        self.detector.reset()
        self.motion.reset()
        self.safety.reset()
        self._history.reset()
        if self.lane_detector:
            self.lane_detector.reset()
        if self._worker is not None:
            self._worker.clear()
        self.latest_vlm = None
        self._latest_vlm_capture_s = None
        self._last_vlm_frame = None
        self._last_signature = None

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
