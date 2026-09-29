"""Dataset format shared by auto-labeling, fine-tuning and evaluation.

labels.jsonl, one record per line (image path relative to the jsonl file):
{
  "image": "frames/clip_000120.jpg",
  "ego_speed_kmh": 50.0,
  "cruise_speed_kmh": 60.0,
  "context": "<SceneContext.summary_text() at that frame>",
  "target": {"longitudinal": "KEEP", "lateral": "KEEP_LANE", "target_speed_kmh": 60, "risk": "low",
             "reason": "..."},
  "split": "train" | "val"           (optional; val must hold out whole videos)
  "reviewed": true                   (optional; set by `adas-vla review`)
}
"""

from __future__ import annotations

import json
from pathlib import Path


def reviews_path(labels: str | Path) -> Path:
    labels = Path(labels)
    return labels.with_name(labels.stem + ".reviews.jsonl")


def splits_path(labels: str | Path) -> Path:
    labels = Path(labels)
    return labels.with_name(labels.stem + ".splits.json")


def load_reviews(labels: str | Path) -> dict[str, dict]:
    """Human reviews from `adas-vla review` (append-only; the last entry per id wins).

    Each review is {"id", "target", "exclude"}; excluded samples are dropped from training and evaluation.
    """
    path = reviews_path(labels)
    reviews = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    reviews[r["id"]] = r
    return reviews


def load_records(path: str | Path, include_excluded: bool = False) -> list[dict]:
    path = Path(path)
    reviews = load_reviews(path)
    split_overlay = json.loads(splits_path(path).read_text()) if splits_path(path).exists() else {}
    records = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("image", "context", "target"):
                if key not in rec:
                    raise ValueError(f"{path}:{line_no}: missing '{key}'")
            rec["image_path"] = str((path.parent / rec["image"]).resolve())
            if rec.get("group") in split_overlay:
                rec["split"] = split_overlay[rec["group"]]
            review = reviews.get(rec.get("id"))
            if review:
                rec["target"] = review["target"]
                rec["reviewed"] = True
                rec["excluded"] = bool(review.get("exclude"))
                if rec["excluded"] and not include_excluded:
                    continue
            rec.setdefault("ego_speed_kmh", 50.0)
            rec.setdefault("cruise_speed_kmh", 60.0)
            records.append(rec)
    return records


def target_json(target: dict) -> str:
    """Canonical, compact serialization of the answer the VLM must learn to produce."""
    keys = ["longitudinal", "lateral", "target_speed_kmh", "risk", "reason"]
    return json.dumps({k: target[k] for k in keys if k in target}, ensure_ascii=False)
