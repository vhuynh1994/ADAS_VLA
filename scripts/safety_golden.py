#!/usr/bin/env python3
"""Generate the safety-gate golden vectors (tests/data/safety_golden.json) from the Python gate.

The file is the equivalence test for any port of the gate (C++ on the SA8797P). Regenerate it only after an
intended behaviour change, and review the diff: every changed expectation is a changed intervention.

    python scripts/safety_golden.py            # rewrite tests/data/safety_golden.json
    python scripts/safety_golden.py --check    # exit 1 if the file no longer matches the gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adas_vla.config import Config  # noqa: E402
from adas_vla.control.golden import config_to_dict, replay  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "safety_golden.json"

VLMS = {
    "none": None,
    "keep": {"longitudinal": "KEEP", "lateral": "KEEP_LANE", "target_speed_kmh": 60},
    "accelerate": {"longitudinal": "ACCELERATE", "lateral": "KEEP_LANE", "target_speed_kmh": 90},
    "decelerate": {"longitudinal": "DECELERATE", "lateral": "KEEP_LANE", "target_speed_kmh": 40},
    "brake": {"longitudinal": "BRAKE", "lateral": "KEEP_LANE", "target_speed_kmh": 20},
    "stop": {"longitudinal": "STOP", "lateral": "KEEP_LANE", "target_speed_kmh": 0},
    "change_left": {"longitudinal": "KEEP", "lateral": "CHANGE_LEFT", "target_speed_kmh": 60},
}


def lead(distance: float, ttc: float | None, cls: str = "car") -> dict:
    return {"cls": cls, "distance_m": distance, "ttc_s": ttc, "closing_speed_mps": None if ttc is None else distance / ttc,
            "in_ego_path": True}


def step(t: float, ego: float, detections: list, vlm: dict | None, lanes: dict | None = None) -> dict:
    return {"t": t, "ego_kmh": ego, "detections": detections, "lanes": lanes,
            "vlm": None if vlm is None else {**vlm, "t": t}}


def scenarios() -> list[dict]:
    cases = []
    objects = {"empty": []}
    for d in (3.0, 10.0, 30.0):
        for ttc in (None, 1.0, 2.5):
            objects[f"car_{d:g}m_ttc{ttc}"] = [lead(d, ttc)]
    for d in (8.0, 20.0):
        objects[f"person_{d:g}m"] = [lead(d, None, "person")]
    objects["cut_in_car_15m"] = [{"cls": "car", "distance_m": 15.0, "ttc_s": None, "closing_speed_mps": 0.0,
                                 "in_ego_path": False, "cutting_in": True}]
    objects["red_light"] = [{"cls": "traffic light", "attribute": "red", "in_ego_path": False}]
    # Each scene is held for 0.2 s: the first step shows an AEB trigger before confirmation (FCW only), the
    # second after `aeb_confirm_s`.
    for ego in (30.0, 60.0):
        for obj_name, dets in objects.items():
            for vlm_name, vlm in VLMS.items():
                cases.append({"name": f"ego{ego:g}_{obj_name}_vlm-{vlm_name}",
                              "steps": [step(10.0, ego, dets, vlm), step(10.2, ego, dets, vlm)]})
    # Stale recommendation -> rule-based ACC policy.
    cases.append({"name": "stale_vlm_uses_rules", "steps": [
        {**step(10.0, 50.0, [lead(30.0, None)], VLMS["keep"]), "vlm": {**VLMS["keep"], "t": 5.0}}]})
    # Lane departure warning, both sides (lanes given at the bottom row, image width 1280).
    cases.append({"name": "ldw_right", "steps": [step(10.0, 50.0, [], VLMS["keep"], {"left_x": 620, "right_x": 1020})]})
    cases.append({"name": "ldw_left", "steps": [step(10.0, 50.0, [], VLMS["keep"], {"left_x": 260, "right_x": 660})]})
    cases.append({"name": "lanes_centered", "steps": [step(10.0, 50.0, [], VLMS["keep"], {"left_x": 440, "right_x": 840})]})
    # AEB confirmation + hold + hysteresis: trigger, confirm, stay on above the trigger threshold, hold through a
    # dropped detection, release (FCW held a little longer).
    cases.append({"name": "aeb_hold_and_hysteresis", "steps": [
        step(10.0, 50.0, [lead(12.0, 1.4)], VLMS["keep"]), step(10.2, 50.0, [lead(11.5, 1.4)], VLMS["keep"]),
        step(10.3, 50.0, [lead(20.0, 1.8)], VLMS["keep"]), step(10.5, 50.0, [], VLMS["keep"]),
        step(10.9, 50.0, [], VLMS["keep"]), step(11.4, 50.0, [], VLMS["keep"]),
    ]})
    # A one-frame phantom never reaches AEB confirmation (FCW acts at once and is held).
    cases.append({"name": "aeb_one_frame_phantom", "steps": [
        step(10.0, 50.0, [lead(3.0, 0.5)], VLMS["keep"]), step(10.05, 50.0, [], VLMS["keep"]),
        step(10.25, 50.0, [], VLMS["keep"]),
    ]})
    # Confirmation counts through a short detection dropout (<= aeb_confirm_gap_s) ...
    cases.append({"name": "aeb_confirm_through_dropout", "steps": [
        step(10.0, 50.0, [lead(5.0, 0.8)], VLMS["keep"]), step(10.05, 50.0, [], VLMS["keep"]),
        step(10.1, 50.0, [lead(4.8, 0.8)], VLMS["keep"]),
    ]})
    # ... but a longer gap restarts it.
    cases.append({"name": "aeb_confirm_restarts_after_gap", "steps": [
        step(10.0, 50.0, [lead(5.0, 0.8)], VLMS["keep"]), step(10.05, 50.0, [], VLMS["keep"]),
        step(10.2, 50.0, [lead(4.8, 0.8)], VLMS["keep"]), step(10.3, 50.0, [lead(4.6, 0.8)], VLMS["keep"]),
    ]})
    # FCW hold: short headway for one frame, then the lead pulls away.
    cases.append({"name": "fcw_hold", "steps": [
        step(10.0, 50.0, [lead(10.0, None)], VLMS["keep"]), step(10.5, 50.0, [lead(40.0, None)], VLMS["keep"]),
        step(11.5, 50.0, [lead(40.0, None)], VLMS["keep"]),
    ]})
    # A VRU stays braked while in path; the gate never relaxes a stronger VLM decision.
    cases.append({"name": "vru_then_clear", "steps": [
        step(10.0, 30.0, [lead(8.0, None, "person")], VLMS["stop"]), step(10.2, 30.0, [], VLMS["keep"])]})
    return cases


def build() -> dict:
    cfg = Config()
    cases = scenarios()
    for case in cases:
        for st, expect in zip(case["steps"], replay(cfg, case)):
            st["expect"] = expect
    return {"config": config_to_dict(cfg), "cases": cases}


def dump(data: dict) -> str:
    """JSON with one line per step, so a regenerated file diffs by intervention."""
    cases = []
    for case in data["cases"]:
        steps = ",\n".join("   " + json.dumps(s, separators=(",", ":")) for s in case["steps"])
        cases.append(f'  {{"name": {json.dumps(case["name"])}, "steps": [\n{steps}\n  ]}}')
    config = json.dumps(data["config"], indent=1).replace("\n", "\n ")
    text = '{\n "config": ' + config + ',\n "cases": [\n' + ",\n".join(cases) + "\n ]\n}\n"
    assert json.loads(text) == data
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = dump(build())

    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} is out of date: run {sys.argv[0]} and review the diff")
        print(f"{OUT}: up to date")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"Wrote {len(build()['cases'])} cases to {OUT}")


if __name__ == "__main__":
    main()
