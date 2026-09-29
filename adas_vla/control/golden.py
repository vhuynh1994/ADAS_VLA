"""Golden test vectors for the safety gate: JSON scenes -> expected arbitration.

`scripts/safety_golden.py` writes tests/data/safety_golden.json from the Python gate; `tests/test_safety_golden.py`
replays it. A port of the gate (C++ for the SA8797P, see docs/DEPLOY_SA8797P.md) must reproduce the same file,
so this module is the reference reader of that format.

Case format (all distances in m, speeds in km/h, times in s):
{"name": "...", "steps": [{"t": 10.0, "ego_kmh": 50, "detections": [{"cls": "car", "distance_m": 12, "ttc_s": 1.0,
  "closing_speed_mps": 12, "in_ego_path": true}], "lanes": {"left_x": 500, "right_x": 900} | null,
  "vlm": {"longitudinal": "KEEP", "lateral": "KEEP_LANE", "target_speed_kmh": 60, "t": 10.0} | null,
  "expect": {"longitudinal": ..., "lateral": ..., "target_speed_kmh": ..., "source": ..., "alerts": [...]}}]}
Steps of one case run through one gate instance in order (they exercise the AEB / FCW hold and hysteresis).
"""

from __future__ import annotations

import dataclasses

from ..config import Config
from ..types import (Alert, Detection, DrivingDecision, EgoState, LaneInfo, LatAction, LongAction, RiskLevel,
                     SceneContext)
from .safety import SafetySupervisor

WIDTH, HEIGHT = 1280, 720


def config_to_dict(cfg: Config) -> dict:
    return {"safety": dataclasses.asdict(cfg.safety), "control": dataclasses.asdict(cfg.control),
            "vlm": {"max_decision_age_s": cfg.vlm.max_decision_age_s}}


def config_from_dict(d: dict) -> Config:
    cfg = Config()
    for section in ("safety", "control"):
        for key, value in d.get(section, {}).items():
            setattr(getattr(cfg, section), key, value)
    cfg.vlm.max_decision_age_s = d.get("vlm", {}).get("max_decision_age_s", cfg.vlm.max_decision_age_s)
    return cfg


def scene_from_dict(step: dict) -> SceneContext:
    detections = [Detection(
        cls_name=o["cls"], conf=1.0, box=tuple(o.get("box", (0.0, 0.0, 0.0, 0.0))), distance_m=o.get("distance_m"),
        closing_speed_mps=o.get("closing_speed_mps"), ttc_s=o.get("ttc_s"), in_ego_path=bool(o.get("in_ego_path")),
        cutting_in=bool(o.get("cutting_in")), attribute=o.get("attribute"),
    ) for o in step.get("detections", [])]
    lanes = LaneInfo(image_width=WIDTH)
    if step.get("lanes"):
        lanes = LaneInfo(left_fit=(0.0, float(step["lanes"]["left_x"])), right_fit=(0.0, float(step["lanes"]["right_x"])),
                         y_top=0.6 * HEIGHT, y_bottom=HEIGHT, image_width=WIDTH)
    return SceneContext(frame_idx=step.get("frame", 0), timestamp_s=step["t"], width=WIDTH, height=HEIGHT,
                        ego=EgoState(step["ego_kmh"]), detections=detections, lanes=lanes)


def decision_from_dict(d: dict | None) -> DrivingDecision | None:
    if d is None:
        return None
    return DrivingDecision(longitudinal=LongAction(d["longitudinal"]), lateral=LatAction(d["lateral"]),
                           target_speed_kmh=float(d["target_speed_kmh"]), risk_level=RiskLevel(d.get("risk", "low")),
                           reason=d.get("reason", ""), source="vlm", timestamp_s=d["t"])


def outcome(decision: DrivingDecision, alerts: list[Alert]) -> dict:
    return {"longitudinal": decision.longitudinal.value, "lateral": decision.lateral.value,
            "target_speed_kmh": round(decision.target_speed_kmh, 3), "source": decision.source,
            "alerts": [a.kind for a in alerts]}


def replay(cfg: Config, case: dict) -> list[dict]:
    gate = SafetySupervisor(cfg)
    return [outcome(*gate.arbitrate(scene_from_dict(step), decision_from_dict(step.get("vlm"))))
            for step in case["steps"]]
