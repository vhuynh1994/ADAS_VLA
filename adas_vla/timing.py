"""Per-stage latency of the per-frame path against the deadline budget (docs/LATENCY_BUDGET.md).

The pipeline stores one `timing` dict (milliseconds) per frame in `FrameResult.timing`; `adas-vla run --log`
writes it to the JSONL log and `adas-vla latency --log` summarizes a log offline (no GPU needed).

Stages (keys of the timing dict):
  read      frame decode / capture (on the SoC: ISP + letterbox)
  detector  YOLO + ByteTrack + box conversion; det_pre / det_model / det_post are the Ultralytics split,
            det_track the rest (tracker + conversion)
  lanes     lane detector; for YOLOP lane_pre / lane_model / lane_post (mask post-processing + line fit)
  geometry  distance / closing speed / TTC / cut-in (MotionEstimator)
  vlm       time the VLM held the frame loop: the whole call in sync mode, the frame copy in async mode
  gate      safety arbitration
  control   controller
  frame     capture -> control command without the VLM (the firm path, deadline = frame period)
  e2e       capture -> control command including a blocking (sync) VLM call
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

SUB_STAGES = ("det_pre", "det_model", "det_post", "det_track", "lane_pre", "lane_model", "lane_post")
STAGES = ("read", "detector", "lanes", "depth", "geometry", "vlm", "gate", "control", "frame", "e2e")


class StageTimer:
    """Wall-clock durations (ms) of the stages of one frame: `with timer("gate"): ...`."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - t0) * 1000)

    def add(self, stage: str, ms: float) -> None:
        self.ms[stage] = self.ms.get(stage, 0.0) + ms

    def update(self, stages: dict[str, float]) -> None:
        for stage, ms in stages.items():
            self.add(stage, ms)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a non-empty list; matches the tail an observer actually saw."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def stage_stats(values: list[float], deadline_ms: float | None = None) -> dict:
    """p50 / p95 / p99 / max of one stage and, given a deadline, the deadline-miss rate (DMR) and the longest run
    of consecutive misses (a firm task tolerates isolated misses, not two in a row)."""
    out = {"n": len(values), "mean": sum(values) / len(values), "p50": percentile(values, 50),
           "p95": percentile(values, 95), "p99": percentile(values, 99), "max": max(values)}
    if deadline_ms is not None:
        misses, run, longest = 0, 0, 0
        for v in values:
            if v > deadline_ms:
                misses += 1
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        out.update(deadline=deadline_ms, misses=misses, dmr=misses / len(values), max_consecutive_misses=longest)
    return out


def summarize(timings: list[dict[str, float]], deadlines_ms: dict[str, float] | None = None,
              warmup: int = 30) -> dict[str, dict]:
    """Per-stage statistics over the frames after `warmup` (model compilation, CUDA kernels, caches).

    Every stage that appears in a frame is reported; the frames that lack a stage (e.g. no lane detector) are not
    counted for it. Stages are ordered as in STAGES, sub-stages and unknown keys after.
    """
    deadlines_ms = deadlines_ms or {}
    series: dict[str, list[float]] = {}
    for t in timings[warmup:]:
        for stage, ms in t.items():
            series.setdefault(stage, []).append(ms)
    order = [s for s in STAGES if s in series] + [s for s in SUB_STAGES if s in series]
    order += sorted(s for s in series if s not in order)
    return {s: stage_stats(series[s], deadlines_ms.get(s)) for s in order}


def ages_summary(ages_s: list[float | None], max_age_s: float, warmup: int = 30) -> dict | None:
    """Age of the VLM decision in effect at each frame: p50 / p95 / max and the share of frames where it was too
    old (or missing) for the gate to use (the gate then falls back to the rule-based policy)."""
    window = ages_s[warmup:]
    if not window:
        return None
    present = [a for a in window if a is not None]
    stale = sum(1 for a in window if a is None or a > max_age_s)
    out = {"n": len(window), "max_age_s": max_age_s, "stale_or_missing": stale / len(window)}
    if present:
        out.update(p50=percentile(present, 50), p95=percentile(present, 95), max=max(present))
    return out


def format_table(stats: dict[str, dict]) -> str:
    """Fixed-width text table of `summarize()` output."""
    lines = [f"{'stage':<11}{'n':>6}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}{'D':>8}{'DMR':>8}{'consec':>7}"]
    for stage, s in stats.items():
        row = f"{stage:<11}{s['n']:>6}{s['p50']:>9.2f}{s['p95']:>9.2f}{s['p99']:>9.2f}{s['max']:>9.2f}"
        if "deadline" in s:
            row += f"{s['deadline']:>8.1f}{100 * s['dmr']:>7.1f}%{s['max_consecutive_misses']:>7d}"
        lines.append(row)
    return "\n".join(lines)


def latency_report(timings: list[dict[str, float]], ages_wall_s: list[float | None], deadlines_ms: dict[str, float],
                   max_age_s: float, warmup: int = 30) -> tuple[str, dict]:
    """Text report + JSON-able dict of a run: per-stage table and, if the VLM ran, the age of its decisions."""
    stats = summarize(timings, deadlines_ms, warmup)
    if not stats:
        return f"No timing after {warmup} warm-up frames.", {}
    text = [f"Latency per stage, ms ({stats[next(iter(stats))]['n']} frames after {warmup} warm-up; "
            f"D = deadline on the SoC, DMR = share of frames over D, consec = longest run of misses):",
            format_table(stats)]
    out = {"warmup": warmup, "stages": stats}
    if any(a is not None for a in ages_wall_s):
        ages = ages_summary(ages_wall_s, max_age_s, warmup)
        out["vlm_age_wall_s"] = ages
        if ages and "p50" in ages:
            text.append(f"VLM decision age (wall clock): p50 {ages['p50']:.2f}s  p95 {ages['p95']:.2f}s  "
                        f"max {ages['max']:.2f}s; older than {max_age_s:g}s or missing in "
                        f"{100 * ages['stale_or_missing']:.1f}% of frames")
    return "\n".join(text), out
