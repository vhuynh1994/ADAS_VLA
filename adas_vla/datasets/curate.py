"""Label curation without a human in the loop: automatic filters, a teacher second opinion, and contact
sheets for visual (AI) review. Every decision is appended to <labels>.reviews.jsonl with its reviewer."""

from __future__ import annotations

import csv
import json
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..training.data import load_records, reviews_path
from .common import risk_for

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def append_reviews(labels: Path, entries: list[dict], reviewer: str) -> int:
    """entries: {"id", "target", "exclude"?, "note"?}. Last entry per id wins when loading."""
    now = datetime.now().isoformat(timespec="seconds")
    with reviews_path(labels).open("a") as f:
        for e in entries:
            f.write(json.dumps({"id": e["id"], "target": e["target"], "exclude": bool(e.get("exclude")),
                                "reviewer": reviewer, "note": e.get("note", ""), "time": now},
                               ensure_ascii=False) + "\n")
    return len(entries)


def exclude_non_ego(labels: Path, metadata_csv: Path) -> int:
    """Australian clips filmed from another vehicle: the ego car is not the one reacting -> unusable."""
    with metadata_csv.open() as f:
        non_ego = {row["video_name"] for row in csv.DictReader(f) if row.get("ego") == "Non-Ego-Centric"}
    entries = [{"id": r["id"], "target": r["target"], "exclude": True, "note": "non ego-centric clip"}
               for r in load_records(labels) if r.get("source") == "australian"
               and Path(r["video"]).name in non_ego]
    return append_reviews(labels, entries, "auto:metadata")


def second_opinion_unexplained_slowdowns(cfg, labels: Path, split: str = "train") -> tuple[int, int]:
    """comma2k19 samples where the driver slowed but perception saw no lead: keep only if the teacher VLM
    also sees a reason to slow down in the image, otherwise exclude (the student cannot learn it)."""
    from PIL import Image as PILImage

    from ..reasoning.vlm import VisionLanguageModel
    from ..types import LongAction

    recs = [r for r in load_records(labels) if r.get("split") == split and not r.get("reviewed")
            and "slowing without visible lead" in (r.get("flags") or [])]
    if not recs:
        return 0, 0
    vlm = VisionLanguageModel(cfg.vlm)
    entries, kept = [], 0
    for r in recs:
        d, raw, _ = vlm.decide(PILImage.open(r["image_path"]), r["context"], r["ego_speed_kmh"],
                               r["cruise_speed_kmh"])
        if d is not None and d.longitudinal.rank >= LongAction.DECELERATE.rank:
            kept += 1
            entries.append({"id": r["id"], "target": r["target"], "note": f"teacher sees reason: {d.reason}"})
        else:
            entries.append({"id": r["id"], "target": r["target"], "exclude": True,
                            "note": "no visible reason to slow down (teacher agrees)"})
    append_reviews(labels, entries, "auto:teacher-7b")
    return kept, len(recs) - kept


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def make_sheets(labels: Path, ids: list[str], out_dir: Path, per_sheet: int = 6, cell_w: int = 560,
                metadata_csv: Path | None = None) -> list[Path]:
    """Grid images (2 columns) with each frame, its proposed label and perception context, numbered."""
    by_id = {r["id"]: r for r in load_records(labels, include_excluded=True)}
    meta = {}
    if metadata_csv and metadata_csv.exists():
        with metadata_csv.open() as f:
            meta = {row["video_name"]: row for row in csv.DictReader(f)}
    out_dir.mkdir(parents=True, exist_ok=True)
    f_big, f_small = _font(17), _font(13)
    sheets, index = [], {}
    for s in range(0, len(ids), per_sheet):
        chunk = ids[s:s + per_sheet]
        cells = []
        for n, sid in enumerate(chunk):
            r = by_id[sid]
            img = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            img = cv2.resize(img, (cell_w, round(h * cell_w / w)))
            t = r["target"]
            header = (f"#{n + 1}  {t['longitudinal']} / {t['lateral']}  {t.get('target_speed_kmh', '?')} km/h"
                      f"   [{r['source']}, ego {r['ego_speed_kmh']:.0f} km/h]")
            ctx_lines = [ln for ln in r["context"].splitlines() if ln and not ln.startswith("Detected objects")]
            video = Path(r.get("video", ""))
            clip = f"{video.parent.name}/{video.name}" if r["source"] != "comma2k19" else video.name
            info = meta.get(video.name, {})
            where = f"clip {clip} frame {r.get('frame')}" + (
                f" ({info['classification']}, hazard {info['hazard']}, {info['location']})" if info else "")
            note = " | ".join(ctx_lines[:4])
            flags = ", ".join(r.get("flags") or [])
            text = textwrap.wrap(where, 78)[:2] + textwrap.wrap(note, 78)[:3] + ([f"flags: {flags}"] if flags else []) + \
                textwrap.wrap("reason: " + t.get("reason", ""), 78)[:1]
            canvas = Image.new("RGB", (cell_w, img.shape[0] + 28 + 17 * len(text)), (24, 24, 28))
            canvas.paste(Image.fromarray(img), (0, 28))
            d = ImageDraw.Draw(canvas)
            d.text((6, 4), header, font=f_big, fill=(255, 220, 90))
            for i, line in enumerate(text):
                d.text((6, img.shape[0] + 30 + 17 * i), line, font=f_small, fill=(220, 220, 220))
            cells.append(np.asarray(canvas))
        cols = 2
        rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
        row_imgs = []
        for row in rows:
            hmax = max(c.shape[0] for c in row)
            padded = [np.pad(c, ((0, hmax - c.shape[0]), (0, 0), (0, 0)), constant_values=40) for c in row]
            if len(padded) < cols:
                padded.append(np.full((hmax, cell_w, 3), 40, np.uint8))
            row_imgs.append(np.concatenate(padded, axis=1))
        sheet = np.concatenate(row_imgs, axis=0)
        path = out_dir / f"sheet_{s // per_sheet + 1:03d}.jpg"
        Image.fromarray(sheet).save(path, quality=88)
        index[path.name] = chunk
        sheets.append(path)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return sheets


def apply_sheet_review(labels: Path, sheets_dir: Path, sheet: str, fixes: dict[int, dict | str],
                       reviewer: str = "claude") -> int:
    """Record a reviewed sheet: numbers not in `fixes` are accepted as-is; a fix is either "exclude" or a
    dict of target fields to change (e.g. {"longitudinal": "BRAKE", "reason": "..."})."""
    index = json.loads((sheets_dir / "index.json").read_text())
    by_id = {r["id"]: r for r in load_records(labels, include_excluded=True)}
    entries = []
    for n, sid in enumerate(index[sheet], start=1):
        target = dict(by_id[sid]["target"])
        fix = fixes.get(n)
        if fix == "exclude":
            entries.append({"id": sid, "target": target, "exclude": True, "note": "excluded in visual review"})
            continue
        if isinstance(fix, dict):
            target.update(fix)
        entries.append({"id": sid, "target": target, "note": "visual review" + (" (fixed)" if fix else "")})
    return append_reviews(labels, entries, reviewer)


def make_filmstrip(labels: Path, video: str, out_path: Path, cols: int = 4, thumb_w: int = 300,
                   metadata_csv: Path | None = None) -> list[str]:
    """All samples of one clip in time order, numbered, for segment-level labeling. Returns the ids."""
    recs = sorted((r for r in load_records(labels, include_excluded=True) if r.get("video") == video),
                  key=lambda r: r["frame"])
    meta = {}
    if metadata_csv and metadata_csv.exists():
        with metadata_csv.open() as f:
            meta = {row["video_name"]: row for row in csv.DictReader(f)}
    info = meta.get(Path(video).name, {})
    f_head, f_small = _font(16), _font(12)
    thumbs = []
    for k, r in enumerate(recs, start=1):
        img = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = cv2.resize(img, (thumb_w, round(h * thumb_w / w)))
        lead = next((ln[2:] for ln in r["context"].splitlines() if "IN EGO LANE" in ln), "")
        canvas = Image.new("RGB", (thumb_w, img.shape[0] + 34), (24, 24, 28))
        canvas.paste(Image.fromarray(img), (0, 18))
        d = ImageDraw.Draw(canvas)
        d.text((4, 1), f"#{k}  frame {r['frame']}", font=f_small, fill=(255, 220, 90))
        d.text((4, img.shape[0] + 19), textwrap.shorten(lead or "no object in ego lane", 46), font=f_small,
               fill=(210, 210, 210))
        thumbs.append(np.asarray(canvas))
    th = max(t.shape[0] for t in thumbs)
    rows = []
    for i in range(0, len(thumbs), cols):
        row = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0)), constant_values=40) for t in thumbs[i:i + cols]]
        row += [np.full((th, thumb_w, 3), 40, np.uint8)] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    r0 = recs[0]
    head = (f"{Path(video).parent.name}/{Path(video).name}  split={r0['split']}  ego assumed {r0['ego_speed_kmh']:.0f} km/h"
            + (f"  | {info['classification']}, {info['ego']}, hazard {info['hazard']}, {info['location']}, "
               f"{info.get('time', '')} {info.get('weather', '')}" if info else ""))
    header = Image.new("RGB", (grid.shape[1], 26), (10, 10, 12))
    ImageDraw.Draw(header).text((6, 4), head, font=f_head, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([np.asarray(header), grid], axis=0)).save(out_path, quality=85)
    return [r["id"] for r in recs]


SPEED_FACTOR = {"KEEP": 1.0, "ACCELERATE": None, "DECELERATE": 0.7, "BRAKE": 0.4, "STOP": 0.0}


def apply_clip_labels(labels: Path, ids: list[str], segments: list[tuple] | str, reviewer: str = "claude") -> int:
    """segments: "exclude", or [(first, last, longitudinal, lateral, reason), ...] over 1-based filmstrip
    numbers; numbers not covered by any segment are excluded (e.g. frames after a crash)."""
    from ..types import LatAction, LongAction

    by_id = {r["id"]: r for r in load_records(labels, include_excluded=True)}
    entries = []
    for k, sid in enumerate(ids, start=1):
        r = by_id[sid]
        seg = None if segments == "exclude" else next((s for s in segments if s[0] <= k <= s[1]), None)
        if seg is None:
            entries.append({"id": sid, "target": r["target"], "exclude": True, "note": "excluded in clip review"})
            continue
        _, _, long_a, lat_a, reason = seg
        ego = r["ego_speed_kmh"]
        factor = SPEED_FACTOR[long_a]
        speed = ego + 10 if factor is None else ego * factor
        target = {"longitudinal": long_a, "lateral": lat_a, "target_speed_kmh": round(speed),
                  "risk": risk_for(LongAction(long_a), LatAction(lat_a)).value, "reason": reason}
        entries.append({"id": sid, "target": target, "note": "clip review"})
    return append_reviews(labels, entries, reviewer)
