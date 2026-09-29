"""Shared helpers for dataset builders: splits, label templates, record writing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..types import LatAction, LongAction, RiskLevel, SceneContext


def split_for(group: str, val_percent: int = 20) -> str:
    """Deterministic train/val split by video group, so frames of one video never straddle splits."""
    bucket = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
    return "val" if bucket < val_percent else "train"


def resplit(labels: Path, val_percent: int = 20, source: str | None = None) -> dict[str, int]:
    """Reassign splits by whole groups (videos/routes) so val holds ~val_percent of the samples.

    Groups are visited in md5 order (deterministic) and added to val until the target is reached. The result
    goes to <labels>.splits.json (group -> split), overlaid by load_records(); labels.jsonl is never rewritten,
    so this is safe while builders are still appending to it.
    """
    from ..training.data import load_records, splits_path

    chosen = [r for r in load_records(labels, include_excluded=True) if source is None or r.get("source") == source]
    sizes: dict[str, int] = {}
    for r in chosen:
        sizes[r["group"]] = sizes.get(r["group"], 0) + 1
    target = val_percent / 100 * len(chosen)
    val_groups, n_val = set(), 0
    for g in sorted(sizes, key=lambda g: hashlib.md5(g.encode()).hexdigest()):
        if n_val >= target:
            break
        if n_val + sizes[g] <= target * 1.25 or not val_groups:
            val_groups.add(g)
            n_val += sizes[g]
    path = splits_path(labels)
    overlay = json.loads(path.read_text()) if path.exists() else {}
    overlay.update({g: ("val" if g in val_groups else "train") for g in sizes})
    path.write_text(json.dumps(overlay, indent=1, sort_keys=True))
    return {"val_groups": len(val_groups), "groups": len(sizes), "val": n_val, "train": len(chosen) - n_val}


def risk_for(long_a: LongAction, lat_a: LatAction) -> RiskLevel:
    if long_a.rank >= LongAction.BRAKE.rank:
        return RiskLevel.HIGH
    if long_a is LongAction.DECELERATE or lat_a in (LatAction.CHANGE_LEFT, LatAction.CHANGE_RIGHT):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def template_reason(long_a: LongAction, lat_a: LatAction, ctx: SceneContext) -> str:
    """Short, factual reason built from the label and perception (no hallucinated details)."""
    lead = ctx.lead_object()
    lead_name = f"lead {lead.cls_name}" if lead is not None else None
    if long_a is LongAction.STOP:
        text = f"{lead_name or 'Traffic'} ahead has stopped; come to a stop."
    elif long_a is LongAction.BRAKE:
        text = f"{lead_name or 'Traffic'} ahead is braking hard; brake firmly."
    elif long_a is LongAction.DECELERATE:
        text = f"Closing on {lead_name}; ease off to keep a safe gap." if lead_name \
            else "Traffic ahead is slowing; ease off."
    elif long_a is LongAction.ACCELERATE:
        text = "Gap ahead is opening; speed up toward cruise speed." if lead_name \
            else "Lane ahead is clear and below cruise speed; speed up."
    else:
        text = f"Following {lead_name} at a steady gap." if lead_name else "Lane ahead is clear; hold speed."
    if lat_a is LatAction.CHANGE_LEFT:
        text = "Changing to the left lane; " + text[0].lower() + text[1:]
    elif lat_a is LatAction.CHANGE_RIGHT:
        text = "Changing to the right lane; " + text[0].lower() + text[1:]
    return text


class RecordWriter:
    """Appends samples to <out>/labels.jsonl and stores frames under <out>/frames/."""

    def __init__(self, out_dir: Path, jpeg_quality: int = 90):
        self.out_dir = out_dir
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "labels.jsonl"
        self.jpeg_quality = jpeg_quality
        self.existing = set()
        if self.path.exists():
            with self.path.open() as f:
                self.existing = {json.loads(line)["id"] for line in f if line.strip()}
        self._f = self.path.open("a")
        self.count = 0

    def has(self, sample_id: str) -> bool:
        return sample_id in self.existing

    def write(self, sample_id: str, frame: np.ndarray, record: dict) -> None:
        name = f"{sample_id}.jpg"
        cv2.imwrite(str(self.frames_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        self._f.write(json.dumps({"id": sample_id, "image": f"frames/{name}", **record}, ensure_ascii=False) + "\n")
        self._f.flush()
        self.existing.add(sample_id)
        self.count += 1

    def close(self) -> None:
        self._f.close()
