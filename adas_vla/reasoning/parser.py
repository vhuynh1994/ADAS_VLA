"""Robust parsing of VLM text output into a DrivingDecision."""

from __future__ import annotations

import json
import re

from ..types import DrivingDecision, LatAction, LongAction, RiskLevel

LONG_ALIASES = {
    "MAINTAIN": LongAction.KEEP, "MAINTAIN_SPEED": LongAction.KEEP, "KEEP_SPEED": LongAction.KEEP,
    "CRUISE": LongAction.KEEP, "CONTINUE": LongAction.KEEP, "FOLLOW": LongAction.KEEP, "HOLD": LongAction.KEEP,
    "SPEED_UP": LongAction.ACCELERATE, "ACCEL": LongAction.ACCELERATE,
    "SLOW_DOWN": LongAction.DECELERATE, "SLOW": LongAction.DECELERATE, "DECEL": LongAction.DECELERATE,
    "YIELD": LongAction.DECELERATE,
    "HARD_BRAKE": LongAction.EMERGENCY_BRAKE, "EMERGENCY_STOP": LongAction.EMERGENCY_BRAKE,
    "HALT": LongAction.STOP, "FULL_STOP": LongAction.STOP,
}
LAT_ALIASES = {
    "KEEP": LatAction.KEEP_LANE, "LANE_KEEP": LatAction.KEEP_LANE, "STRAIGHT": LatAction.KEEP_LANE,
    "GO_STRAIGHT": LatAction.KEEP_LANE, "NONE": LatAction.KEEP_LANE, "MAINTAIN": LatAction.KEEP_LANE,
    "CHANGE_LANE_LEFT": LatAction.CHANGE_LEFT, "LANE_CHANGE_LEFT": LatAction.CHANGE_LEFT,
    "CHANGE_LANE_RIGHT": LatAction.CHANGE_RIGHT, "LANE_CHANGE_RIGHT": LatAction.CHANGE_RIGHT,
    "STEER_LEFT": LatAction.NUDGE_LEFT, "STEER_RIGHT": LatAction.NUDGE_RIGHT,
}
# Legacy single-action outputs (9-action schema) mapped onto the two axes.
LEGACY_ACTIONS = {
    "NUDGE_LEFT": (LongAction.KEEP, LatAction.NUDGE_LEFT), "NUDGE_RIGHT": (LongAction.KEEP, LatAction.NUDGE_RIGHT),
    "CHANGE_LANE_LEFT": (LongAction.KEEP, LatAction.CHANGE_LEFT),
    "CHANGE_LANE_RIGHT": (LongAction.KEEP, LatAction.CHANGE_RIGHT),
}


def extract_json(text: str) -> dict | None:
    """Return the first balanced JSON object found in `text`, tolerating fences and trailing commas."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
                        try:
                            obj = json.loads(attempt)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                    break
        start = text.find("{", start + 1)
    return None


def _key(value) -> str | None:
    return re.sub(r"[\s\-]+", "_", value.strip().upper()) if isinstance(value, str) else None


def normalize_long(value) -> LongAction | None:
    key = _key(value)
    if key is None:
        return None
    try:
        return LongAction(key)
    except ValueError:
        return LONG_ALIASES.get(key)


def normalize_lat(value) -> LatAction | None:
    key = _key(value)
    if key is None:
        return None
    try:
        return LatAction(key)
    except ValueError:
        return LAT_ALIASES.get(key)


def _as_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(m.group()) if m else None
    return None


def parse_decision(text: str, ego_speed_kmh: float, max_speed_kmh: float = 130.0) -> DrivingDecision | None:
    """Parse VLM output. Returns None if no valid longitudinal action can be recovered."""
    obj = extract_json(text)
    if obj is None:
        return None
    long_a = normalize_long(obj.get("longitudinal"))
    lat_a = normalize_lat(obj.get("lateral")) if "lateral" in obj else LatAction.KEEP_LANE
    if long_a is None and "action" in obj:  # legacy single-action schema
        key = _key(obj["action"])
        long_a, lat_a = LEGACY_ACTIONS.get(key, (normalize_long(obj["action"]), LatAction.KEEP_LANE))
    if long_a is None:
        return None
    lat_a = lat_a or LatAction.KEEP_LANE

    speed = _as_float(obj.get("target_speed_kmh"))
    if speed is None:
        speed = {
            LongAction.ACCELERATE: ego_speed_kmh + 10, LongAction.DECELERATE: ego_speed_kmh * 0.8,
            LongAction.BRAKE: ego_speed_kmh * 0.5, LongAction.STOP: 0.0, LongAction.EMERGENCY_BRAKE: 0.0,
        }.get(long_a, ego_speed_kmh)
    speed = min(max(speed, 0.0), max_speed_kmh)

    try:
        risk = RiskLevel(str(obj.get("risk", obj.get("risk_level", "medium"))).strip().lower())
    except ValueError:
        risk = RiskLevel.MEDIUM

    return DrivingDecision(
        longitudinal=long_a, lateral=lat_a, target_speed_kmh=speed, risk_level=risk,
        reason=str(obj.get("reason", "")).strip(), source="vlm",
    )
