"""Extract ADAS intervention events from a `adas-vla run --log` JSONL file."""

from __future__ import annotations

import json
from pathlib import Path

EVENT_ALERTS = {"AEB", "FCW", "PED", "LDW", "RED_LIGHT", "LANE_CHANGE_SUGGESTED"}


def load_run_log(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


class EventDetector:
    """Online version of extract_events(): returns the new event kinds of each frame (debounced)."""

    def __init__(self, min_gap_s: float = 2.0):
        self.min_gap_s = min_gap_s
        self._last_seen: dict[str, float] = {}

    def update(self, t: float, alert_kinds: list[str], decision_source: str) -> list[str]:
        kinds = {k for k in alert_kinds if k in EVENT_ALERTS}
        if decision_source == "safety":
            kinds.add("SAFETY_OVERRIDE")
        new = sorted(k for k in kinds if t - self._last_seen.get(k, -1e9) > self.min_gap_s)
        for k in kinds:
            self._last_seen[k] = t
        return new


def frame_record(res) -> dict:
    """The per-frame log record written by `adas-vla run --log` (also the LLM's event input)."""
    return {
        "frame": res.context.frame_idx, "t": round(res.context.timestamp_s, 3),
        "decision": res.decision.to_dict(), "alerts": [a.__dict__ for a in res.alerts],
        "command": {k: (v.value if hasattr(v, "value") else round(v, 3)) for k, v in res.command.__dict__.items()},
        "objects": [d.describe() for d in res.context.detections],
        "timing": {k: round(v, 3) for k, v in res.timing.items()},
        "vlm_age_s": None if res.vlm_age_s is None else round(res.vlm_age_s, 3),
        "vlm_age_wall_s": None if res.vlm_age_wall_s is None else round(res.vlm_age_wall_s, 3),
    }


def extract_events(frames: list[dict], min_gap_s: float = 2.0) -> list[dict]:
    """One event per episode: the first frame of each alert kind (or safety override),
    ignoring repeats of the same kind within `min_gap_s`."""
    events, last_seen = [], {}
    for fr in frames:
        kinds = {a["kind"] for a in fr.get("alerts", []) if a["kind"] in EVENT_ALERTS}
        if fr["decision"].get("source") == "safety":
            kinds.add("SAFETY_OVERRIDE")
        new = {k for k in kinds if fr["t"] - last_seen.get(k, -1e9) > min_gap_s}
        for k in kinds:
            last_seen[k] = fr["t"]
        if new:
            events.append({**fr, "kinds": sorted(new)})
    return events
