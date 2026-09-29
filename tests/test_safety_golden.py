"""The safety gate must reproduce tests/data/safety_golden.json (also the equivalence test for a C++ port)."""

import json
from pathlib import Path

from adas_vla.control.golden import config_from_dict, replay

GOLDEN = Path(__file__).parent / "data" / "safety_golden.json"


def test_safety_gate_matches_golden_vectors():
    data = json.loads(GOLDEN.read_text())
    cfg = config_from_dict(data["config"])
    assert len(data["cases"]) > 150
    changed = []
    for case in data["cases"]:
        got = replay(cfg, case)
        expected = [step["expect"] for step in case["steps"]]
        if got != expected:
            changed.append((case["name"], expected, got))
    assert not changed, (f"{len(changed)} golden cases changed, first: {changed[0]}. "
                         "After an intended change: python scripts/safety_golden.py, then review the diff.")
