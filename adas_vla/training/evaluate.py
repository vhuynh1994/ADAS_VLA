"""Offline evaluation: JSON validity, per-axis and joint action accuracy, under-braking, speed error, latency."""

from __future__ import annotations

import json
from collections import Counter

from PIL import Image

from ..config import Config
from ..reasoning.parser import parse_decision
from ..reasoning.vlm import VisionLanguageModel
from ..types import LongAction
from .data import load_records


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
    speed_errors, latencies = [], []
    conf_long: Counter = Counter()
    conf_lat: Counter = Counter()
    report = open(report_path, "w") if report_path else None

    for rec in records:
        gt = parse_decision(json.dumps(rec["target"]), rec["ego_speed_kmh"], cfg.safety.max_speed_kmh)
        if gt is None:
            print(f"skip {rec['image']}: invalid label {rec['target']}")
            continue
        pred, raw, latency = vlm.decide(Image.open(rec["image_path"]), rec["context"], rec["ego_speed_kmh"],
                                        rec["cruise_speed_kmh"], cfg.safety.max_speed_kmh)
        n += 1
        latencies.append(latency)
        if pred is not None:
            valid += 1
            long_ok += pred.longitudinal is gt.longitudinal
            lat_ok += pred.lateral is gt.lateral
            joint_ok += pred.longitudinal is gt.longitudinal and pred.lateral is gt.lateral
            # Safety-relevant error: the label asks to slow down or brake, the prediction brakes less.
            under_brake += (gt.longitudinal.rank >= LongAction.DECELERATE.rank
                            and pred.longitudinal.rank < gt.longitudinal.rank)
            speed_errors.append(abs(pred.target_speed_kmh - gt.target_speed_kmh))
        conf_long[(gt.longitudinal.value, pred.longitudinal.value if pred else "INVALID")] += 1
        conf_lat[(gt.lateral.value, pred.lateral.value if pred else "INVALID")] += 1
        if report:
            report.write(json.dumps({
                "image": rec["image"], "gt": gt.target_dict(), "pred": pred.target_dict() if pred else None,
                "probs": pred.action_probs if pred else None,
                "latency_s": round(latency, 3), "raw": raw}, ensure_ascii=False) + "\n")
    if report:
        report.close()

    lat = sorted(latencies)
    metrics = {
        "samples": n,
        "json_valid_rate": valid / max(1, n),
        "joint_accuracy": joint_ok / max(1, n),
        "longitudinal_accuracy": long_ok / max(1, n),
        "lateral_accuracy": lat_ok / max(1, n),
        "under_braking_rate": under_brake / max(1, n),
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
