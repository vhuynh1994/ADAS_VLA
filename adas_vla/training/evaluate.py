"""Offline evaluation: JSON validity, per-axis and joint action accuracy, under-braking (of the VLM alone and
after the safety gate), speed error, latency, with 95% confidence intervals."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace

from ..config import Config
from ..control import SafetySupervisor
from ..reasoning.parser import parse_decision
from ..reasoning.vlm import VisionLanguageModel
from ..types import Detection, DrivingDecision, EgoState, LaneInfo, LongAction, SceneContext
from .data import load_records, sample_visual


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a rate k / n, as (low, high) in 0..1."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def under_brakes(gt: LongAction, pred: LongAction) -> bool:
    """Safety-relevant error: the label asks to slow down or brake, the prediction brakes less."""
    return gt.rank >= LongAction.DECELERATE.rank and pred.rank < gt.rank


def gate_offline(cfg: Config, rec: dict, decision: DrivingDecision) -> DrivingDecision:
    """Replay the safety gate on a logged sample. The record's stored lead object (`lead`, written by the dataset
    builders) and ego speed rebuild the scene the gate needs; the sample's own cruise speed sets the envelope."""
    lead = rec.get("lead")
    detections = []
    if lead:
        detections = [Detection(
            cls_name=lead["cls"], conf=1.0, box=(0.0, 0.0, 0.0, 0.0), distance_m=lead.get("distance_m"),
            closing_speed_mps=lead.get("closing_speed_mps"), ttc_s=lead.get("ttc_s"),
            in_ego_path=not lead.get("cutting_in"), cutting_in=bool(lead.get("cutting_in")),
        )]
    ctx = SceneContext(frame_idx=0, timestamp_s=0.0, width=1280, height=720, ego=EgoState(rec["ego_speed_kmh"]),
                       detections=detections, lanes=LaneInfo(image_width=1280))
    cfg = replace(cfg, control=replace(cfg.control, cruise_speed_kmh=rec.get("cruise_speed_kmh", 60.0)))
    final, _ = SafetySupervisor(cfg).arbitrate(ctx, replace(decision, timestamp_s=0.0))
    return final


def select_records(path: str, split: str | None = None, reviewed_only: bool = False,
                   limit: int | None = None) -> list[dict]:
    records = load_records(path)
    if split:
        records = [r for r in records if r.get("split", "train") == split]
    if reviewed_only:
        records = [r for r in records if r.get("reviewed")]
    return records[:limit] if limit else records


def evaluate(cfg: Config, data: str, limit: int | None = None, report_path: str | None = None,
             split: str | None = None, reviewed_only: bool = False, vlm: VisionLanguageModel | None = None) -> dict:
    records = select_records(data, split, reviewed_only, limit)
    if not records:
        raise SystemExit(f"No records in {data} (split={split}, reviewed_only={reviewed_only})")
    vlm = vlm or VisionLanguageModel(cfg.vlm)
    n = valid = long_ok = lat_ok = joint_ok = under_brake = 0
    replayed = system_under_brake = 0
    speed_errors, latencies = [], []
    conf_long: Counter = Counter()
    conf_lat: Counter = Counter()
    report = open(report_path, "w") if report_path else None

    for rec in records:
        gt = parse_decision(json.dumps(rec["target"]), rec["ego_speed_kmh"], cfg.safety.max_speed_kmh)
        if gt is None:
            print(f"skip {rec['image']}: invalid label {rec['target']}")
            continue
        visual = sample_visual(rec, cfg)
        image, prev = (visual[1], visual[0]) if isinstance(visual, list) else (visual, None)
        pred, raw, latency = vlm.decide(image, rec["context"], rec["ego_speed_kmh"], rec["cruise_speed_kmh"],
                                        cfg.safety.max_speed_kmh, prev_image=prev)
        n += 1
        latencies.append(latency)
        final = None
        if pred is not None:
            valid += 1
            long_ok += pred.longitudinal is gt.longitudinal
            lat_ok += pred.lateral is gt.lateral
            joint_ok += pred.longitudinal is gt.longitudinal and pred.lateral is gt.lateral
            under_brake += under_brakes(gt.longitudinal, pred.longitudinal)
            speed_errors.append(abs(pred.target_speed_kmh - gt.target_speed_kmh))
            if "lead" in rec:  # the system = VLM + safety gate, replayed on the stored perception
                final = gate_offline(cfg, rec, pred)
                replayed += 1
                system_under_brake += under_brakes(gt.longitudinal, final.longitudinal)
        conf_long[(gt.longitudinal.value, pred.longitudinal.value if pred else "INVALID")] += 1
        conf_lat[(gt.lateral.value, pred.lateral.value if pred else "INVALID")] += 1
        if report:
            report.write(json.dumps({
                "image": rec["image"], "gt": gt.target_dict(), "pred": pred.target_dict() if pred else None,
                "final": final.target_dict() if final else None, "probs": pred.action_probs if pred else None,
                "latency_s": round(latency, 3), "raw": raw}, ensure_ascii=False) + "\n")
    if report:
        report.close()

    lat = sorted(latencies)
    metrics = {
        "samples": n,
        "json_valid_rate": valid / max(1, n),
        "joint_accuracy": joint_ok / max(1, n),
        "joint_accuracy_ci95": wilson_interval(joint_ok, n),
        "longitudinal_accuracy": long_ok / max(1, n),
        "lateral_accuracy": lat_ok / max(1, n),
        "under_braking_rate": under_brake / max(1, n),
        "under_braking_ci95": wilson_interval(under_brake, n),
        "system_under_braking_rate": system_under_brake / replayed if replayed else None,
        "replayed_samples": replayed,
        "target_speed_mae_kmh": sum(speed_errors) / max(1, len(speed_errors)),
        "latency_p50_s": lat[len(lat) // 2] if lat else 0.0,
        "latency_p90_s": lat[min(len(lat) - 1, int(len(lat) * 0.9))] if lat else 0.0,
    }
    print(json.dumps(metrics, indent=2))
    for name, conf in (("longitudinal", conf_long), ("lateral", conf_lat)):
        print(f"\nConfusion {name} (ground truth -> prediction):")
        for (g, p), count in sorted(conf.items(), key=lambda kv: -kv[1]):
            print(f"  {g:<16} -> {p:<16} {count}")
    return metrics
