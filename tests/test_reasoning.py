from adas_vla.reasoning.parser import extract_json, normalize_lat, normalize_long, parse_decision
from adas_vla.reasoning.prompts import chat_messages, decision_messages
from adas_vla.types import LatAction, LongAction, RiskLevel


def test_extract_json_with_fences_and_prose():
    text = 'Sure! ```json\n{"action": "BRAKE", "reason": "car {ahead}"}\n``` done'
    assert extract_json(text) == {"action": "BRAKE", "reason": "car {ahead}"}


def test_extract_json_trailing_comma_and_second_object():
    assert extract_json('garbage {not json} then {"a": [1, 2,],}') == {"a": [1, 2]}


def test_extract_json_none():
    assert extract_json("no json here") is None


def test_normalize_aliases():
    assert normalize_long("maintain speed") is LongAction.KEEP
    assert normalize_long("slow-down") is LongAction.DECELERATE
    assert normalize_long("fly") is None and normalize_long(None) is None
    assert normalize_lat("keep lane") is LatAction.KEEP_LANE
    assert normalize_lat("lane change left") is LatAction.CHANGE_LEFT


def test_parse_decision_full():
    raw = ('{"longitudinal": "decelerate", "lateral": "KEEP_LANE", "target_speed_kmh": "45 km/h",'
           ' "risk": "Medium", "reason": "slower car"}')
    d = parse_decision(raw, ego_speed_kmh=60)
    assert d.longitudinal is LongAction.DECELERATE and d.lateral is LatAction.KEEP_LANE
    assert d.target_speed_kmh == 45 and d.risk_level is RiskLevel.MEDIUM and d.reason == "slower car"


def test_parse_decision_defaults_and_clamping():
    d = parse_decision('{"longitudinal": "ACCELERATE", "target_speed_kmh": 400}', ego_speed_kmh=50, max_speed_kmh=130)
    assert d.target_speed_kmh == 130 and d.lateral is LatAction.KEEP_LANE and d.risk_level is RiskLevel.MEDIUM
    assert parse_decision('{"longitudinal": "STOP"}', ego_speed_kmh=50).target_speed_kmh == 0


def test_parse_decision_legacy_single_action():
    d = parse_decision('{"action": "NUDGE_LEFT", "risk_level": "low"}', 60)
    assert d.longitudinal is LongAction.KEEP and d.lateral is LatAction.NUDGE_LEFT and d.risk_level is RiskLevel.LOW
    assert parse_decision('{"action": "MAINTAIN"}', 60).longitudinal is LongAction.KEEP


def test_parse_decision_invalid_action():
    assert parse_decision('{"longitudinal": "DANCE"}', 50) is None


def test_prefixed_output_round_trip():
    from adas_vla.reasoning.prompts import DECISION_PREFIX
    from adas_vla.training.data import target_json

    target = {"longitudinal": "BRAKE", "lateral": "KEEP_LANE", "target_speed_kmh": 20, "risk": "high",
              "reason": "pedestrian crossing"}
    text = target_json(target)
    assert text.startswith(DECISION_PREFIX)  # training targets start exactly like forced inference output
    assert parse_decision(text, 40).target_dict() == target


def test_decision_messages_structure():
    msgs = decision_messages("IMG", "Lane: ok.", 50, 60, language="vi")
    assert msgs[0]["role"] == "system"
    content = msgs[1]["content"]
    assert content[0] == {"type": "image", "image": "IMG"}
    assert "Vietnamese" in content[1]["text"]
    assert "EMERGENCY_BRAKE" not in content[1]["text"]  # reserved for the safety gate
    assert "CHANGE_LEFT" in content[1]["text"] and "DECELERATE" in content[1]["text"]


def test_chat_messages_image_only_first_turn():
    msgs = chat_messages("IMG", "ctx", 50, [("q1", "a1")], "q2")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"][0]["type"] == "image"
    assert all(c["type"] == "text" for c in msgs[3]["content"])


def test_extract_events_debounces_and_flags_safety():
    from adas_vla.events import extract_events

    def fr(t, kinds=(), source="rules"):
        return {"t": t, "frame": int(t * 10), "alerts": [{"kind": k, "level": "warning", "message": k} for k in kinds],
                "decision": {"longitudinal": "KEEP", "lateral": "KEEP_LANE", "source": source,
                             "target_speed_kmh": 60}}

    frames = [fr(0.0), fr(1.0, ["FCW"]), fr(1.1, ["FCW"]), fr(1.2, ["FCW", "AEB"], "safety"),
              fr(5.0, ["FCW"]), fr(5.1, ["INFO_ONLY"])]
    events = extract_events(frames, min_gap_s=2.0)
    assert [e["t"] for e in events] == [1.0, 1.2, 5.0]
    assert events[1]["kinds"] == ["AEB", "SAFETY_OVERRIDE"]


def test_event_prompt_contains_facts():
    from adas_vla.reasoning.llm import event_prompt

    ev = {"t": 3.2, "frame": 80, "alerts": [{"kind": "AEB", "level": "critical", "message": "car at 6 m"}],
          "decision": {"longitudinal": "EMERGENCY_BRAKE", "lateral": "KEEP_LANE", "source": "safety",
                       "target_speed_kmh": 0, "reason": "AEB"},
          "objects": ["car, ahead, ~6 m, IN EGO LANE"]}
    text = event_prompt(ev)
    assert "EMERGENCY_BRAKE" in text and "car at 6 m" in text and "IN EGO LANE" in text


def test_alerts_text_grounds_copilot():
    from adas_vla.reasoning.prompts import CHAT_SYSTEM_PROMPT, alerts_text
    from adas_vla.types import Alert

    assert "none" in alerts_text([])
    text = alerts_text([Alert("AEB", "Emergency brake: person at 3 m", "critical")])
    assert "AEB (critical): Emergency brake: person at 3 m" in text
    assert "never contradict" in CHAT_SYSTEM_PROMPT


def test_balanced_epoch_caps_dominant_label():
    import random

    from adas_vla.training.finetune import balanced_epoch, label_key

    recs = [{"target": {"longitudinal": "KEEP", "lateral": "KEEP_LANE"}}] * 900 + \
           [{"target": {"longitudinal": "DECELERATE", "lateral": "KEEP_LANE"}}] * 80 + \
           [{"target": {"longitudinal": "BRAKE", "lateral": "KEEP_LANE"}}] * 20
    order = balanced_epoch(recs, 0.35, random.Random(0))
    keep = sum(label_key(recs[i])[0] == "KEEP" for i in order)
    assert keep <= 0.35 * len(order) + 1
    assert sum(label_key(recs[i])[0] == "BRAKE" for i in order) == 20  # minority kept in full
    assert len(set(order)) == len(order)


def test_reviews_overlay_and_exclusion(tmp_path):
    import json

    from adas_vla.training.data import load_records, reviews_path

    labels = tmp_path / "labels.jsonl"
    base = {"image": "f.jpg", "context": "c", "target": {"longitudinal": "KEEP", "lateral": "KEEP_LANE"}}
    labels.write_text("".join(json.dumps({"id": i, **base}) + "\n" for i in ("a", "b", "c")))
    fixed = {"longitudinal": "BRAKE", "lateral": "KEEP_LANE", "target_speed_kmh": 0, "risk": "high", "reason": "x"}
    reviews_path(labels).write_text(
        json.dumps({"id": "a", "target": {**fixed, "longitudinal": "DECELERATE"}}) + "\n"
        + json.dumps({"id": "a", "target": fixed}) + "\n"  # last review wins
        + json.dumps({"id": "b", "target": base["target"], "exclude": True}) + "\n")
    recs = {r["id"]: r for r in load_records(labels)}
    assert set(recs) == {"a", "c"} and recs["a"]["target"]["longitudinal"] == "BRAKE" and recs["a"]["reviewed"]
    assert len(load_records(labels, include_excluded=True)) == 3


def test_resplit_writes_overlay_without_touching_labels(tmp_path):
    import json

    from adas_vla.datasets.common import resplit
    from adas_vla.training.data import load_records

    labels = tmp_path / "labels.jsonl"
    rows = [{"id": f"{g}_{i}", "image": "f.jpg", "context": "c", "group": g, "source": "comma2k19", "split": "val",
             "target": {"longitudinal": "KEEP", "lateral": "KEEP_LANE"}} for g in "abcdefghij" for i in range(10)]
    labels.write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = labels.read_text()
    stats = resplit(labels, val_percent=20)
    assert labels.read_text() == before  # never rewritten
    recs = load_records(labels)
    val_groups = {r["group"] for r in recs if r["split"] == "val"}
    assert stats["val"] == 20 and len(val_groups) == 2
    assert all(r["split"] == ("val" if r["group"] in val_groups else "train") for r in recs)


def test_close_before_reason():
    from adas_vla.reasoning.vlm import close_before_reason

    raw = '{"longitudinal": "BRAKE", "lateral": "KEEP_LANE", "target_speed_kmh": 20, "risk": "high", "reason"'
    fixed = close_before_reason(raw)
    assert fixed.endswith('"high"}')
    d = parse_decision(fixed, 40)
    assert d.longitudinal is LongAction.BRAKE and d.reason == ""
    full = raw + ': "x"}'
    assert close_before_reason('{"a": 1}') == '{"a": 1}'
    assert parse_decision(full, 40).reason == "x"


def test_choose_action_cautious_escalates_but_never_relaxes():
    from adas_vla.reasoning.vlm import choose_action

    p = {"ACCELERATE": 0.05, "KEEP": 0.55, "DECELERATE": 0.25, "BRAKE": 0.10, "STOP": 0.05}
    assert choose_action(p, "greedy", 0.35, 0.3) == "KEEP"
    assert choose_action(p, "cautious", 0.35, 0.3) == "DECELERATE"  # P(>= DECELERATE) = 0.40
    assert choose_action(p, "cautious", 0.45, 0.3) == "KEEP"
    p2 = {"ACCELERATE": 0.0, "KEEP": 0.5, "DECELERATE": 0.15, "BRAKE": 0.25, "STOP": 0.1}
    assert choose_action(p2, "cautious", 0.35, 0.3) == "BRAKE"  # P(>= BRAKE) = 0.35
    p3 = {"ACCELERATE": 0.0, "KEEP": 0.1, "DECELERATE": 0.1, "BRAKE": 0.1, "STOP": 0.7}
    assert choose_action(p3, "cautious", 0.35, 0.3) == "STOP"  # never relaxed
