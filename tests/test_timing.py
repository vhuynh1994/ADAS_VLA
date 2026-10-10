import time

import numpy as np
import pytest

from adas_vla.config import Config, load_config
from adas_vla.timing import StageTimer, ages_summary, latency_report, percentile, stage_stats, summarize
from adas_vla.types import DrivingDecision, EgoState, LatAction, LongAction


def test_percentile_nearest_rank():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile(values, 99) == 99
    assert percentile(values, 100) == 100
    assert percentile([7.0], 99) == 7.0


def test_stage_stats_deadline_misses_and_longest_run():
    s = stage_stats([1, 5, 6, 1, 7, 8, 9, 1], deadline_ms=4)
    assert s["misses"] == 5 and s["dmr"] == pytest.approx(5 / 8)
    assert s["max_consecutive_misses"] == 3
    assert s["max"] == 9 and s["n"] == 8
    assert "dmr" not in stage_stats([1, 2])  # no deadline -> no miss statistics


def test_summarize_skips_warmup_and_orders_stages():
    timings = [{"detector": 100.0, "gate": 0.1}] * 3 + [{"gate": 0.2, "detector": 10.0, "det_model": 6.0}] * 10
    stats = summarize(timings, {"detector": 15.0, "gate": 1.0}, warmup=3)
    assert list(stats) == ["detector", "gate", "det_model"]
    assert stats["detector"]["max"] == 10.0 and stats["detector"]["dmr"] == 0
    assert stats["det_model"]["n"] == 10 and "deadline" not in stats["det_model"]
    assert summarize(timings, warmup=100) == {}


def test_ages_summary_counts_stale_and_missing():
    ages = [None, None] + [0.5] * 6 + [3.0, 2.5]
    a = ages_summary(ages, max_age_s=2.0, warmup=0)
    assert a["stale_or_missing"] == pytest.approx(4 / 10)
    assert a["p50"] == 0.5 and a["max"] == 3.0
    text, out = latency_report([{"gate": 0.1}] * 10, ages, {"gate": 1.0}, 2.0, warmup=0)
    assert "VLM decision age" in text and out["vlm_age_wall_s"]["n"] == 10


def test_stage_timer_accumulates():
    timer = StageTimer()
    with timer("gate"):
        time.sleep(0.001)
    timer.add("gate", 1.0)
    timer.update({"det_model": 2.0})
    assert timer.ms["gate"] > 1.5 and timer.ms["det_model"] == 2.0


def test_budget_partial_override_keeps_other_deadlines():
    cfg = load_config(overrides=["budget.deadlines_ms.gate=2.0"])
    assert cfg.budget.deadlines_ms["gate"] == 2.0
    assert cfg.budget.deadlines_ms["detector"] == 15.0 and cfg.budget.deadlines_ms["frame"] == 33.3


class _FakeDetector:
    def __init__(self, cfg):
        self.last_timing = {}

    def __call__(self, frame):
        self.last_timing = {"det_pre": 0.1, "det_model": 0.2, "det_post": 0.1}
        return []

    def reset(self):
        pass


class _FakeVLM:
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.calls = 0

    def decide(self, frame, context_text, ego_kmh, cruise_kmh, max_kmh, prev_image=None):
        self.calls += 1
        time.sleep(self.delay_s)
        return DrivingDecision(LongAction.KEEP, LatAction.KEEP_LANE, 50.0), "{}", self.delay_s


def _pipeline(monkeypatch, mode="sync", delay_s=0.0):
    import adas_vla.perception.detector as detector_module
    from adas_vla.pipeline import ADASPipeline

    monkeypatch.setattr(detector_module, "ObjectDetector", _FakeDetector)
    cfg = Config()
    cfg.perception.lane_model = "classic"
    cfg.vlm.mode, cfg.vlm.trigger, cfg.vlm.every_n_frames = mode, "interval", 3
    return ADASPipeline(cfg, vlm=_FakeVLM(delay_s))


def test_pipeline_logs_stage_timing_and_decision_age(monkeypatch):
    from adas_vla.events import frame_record

    pipe = _pipeline(monkeypatch)
    frame = np.full((360, 640, 3), 60, np.uint8)
    results = [pipe.process(frame, i, i / 30, EgoState(60), read_ms=2.0) for i in range(5)]
    t = results[-1].timing
    for stage in ("read", "detector", "det_model", "lanes", "geometry", "vlm", "gate", "control", "frame", "e2e"):
        assert stage in t, stage
    assert t["read"] == 2.0 and t["frame"] <= t["e2e"] and t["e2e"] >= 2.0
    # VLM called on frames 0 and 3 (every 3 frames): age on the frame timeline
    assert [round(r.vlm_age_s * 30) for r in results] == [0, 1, 2, 0, 1]
    assert all(r.vlm_age_wall_s >= 0 for r in results)
    rec = frame_record(results[-1])
    assert rec["timing"]["gate"] >= 0 and rec["vlm_age_s"] == pytest.approx(1 / 30, abs=1e-3)


def test_async_reset_drops_decisions_of_the_previous_video(monkeypatch):
    pipe = _pipeline(monkeypatch, mode="async", delay_s=0.05)
    frame = np.full((360, 640, 3), 60, np.uint8)
    try:
        pipe.process(frame, 0, 0.0, EgoState(60))
        pipe.reset()  # call still running: its result belongs to the old video
        time.sleep(0.2)
        res = pipe.process(frame, 1, 0.0, EgoState(60))
        assert res.vlm_decision is None and res.vlm_age_s is None
        time.sleep(0.2)
        res = pipe.process(frame, 2, 0.1, EgoState(60))
        assert res.vlm_decision is not None and res.vlm_age_wall_s >= 0.05
    finally:
        pipe.close()


def _on_result(decision, raw, latency, ctx):
    decision.frame_idx, decision.timestamp_s = ctx.frame_idx, ctx.timestamp_s
    return decision


def test_process_worker_runs_the_newest_frame_and_drops_old_generations():
    from adas_vla.pipeline import _ProcessVLMWorker
    from adas_vla.types import LaneInfo, SceneContext

    def ctx(i):
        return SceneContext(i, i / 30, 640, 360, EgoState(60), [], LaneInfo(image_width=640))

    frame = np.zeros((36, 64, 3), np.uint8)
    worker = _ProcessVLMWorker(_SlowFakeVLM, 60.0, 130.0, _on_result)
    try:
        worker.submit((frame, None), ctx(0), 1.0)  # sent at once
        worker.submit((frame, None), ctx(1), 2.0)  # child busy: pending
        worker.submit((frame, None), ctx(2), 3.0)  # replaces frame 1
        deadline = time.monotonic() + 10
        seen = []
        while time.monotonic() < deadline and (not seen or seen[-1] != 2):
            latest = worker.latest()
            if latest and (not seen or seen[-1] != latest[0].frame_idx):
                seen.append(latest[0].frame_idx)
            time.sleep(0.01)
        assert seen == [0, 2] and worker.latest()[1] == 3.0
        worker.submit((frame, None), ctx(3), 4.0)
        worker.clear()  # result of frame 3 belongs to the previous video
        time.sleep(0.5)
        assert worker.latest() is None
    finally:
        worker.close()


class _SlowFakeVLM(_FakeVLM):
    def __init__(self):
        super().__init__(delay_s=0.1)
