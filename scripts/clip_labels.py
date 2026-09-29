#!/usr/bin/env python3
"""Apply segment labels written during visual clip review: {strip: "exclude" | [[first, last, long, lat, reason]...]}."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adas_vla.datasets.curate import apply_clip_labels  # noqa: E402

labels, strips_dir, batch = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
index = json.loads((strips_dir / "index.json").read_text())
n = 0
for strip, segs in batch.items():
    ids = index[strip]["ids"]
    if segs != "exclude":
        segs = [tuple(s) for s in segs]
        assert all(1 <= s[0] <= s[1] <= len(ids) for s in segs), (strip, segs, len(ids))
    n += apply_clip_labels(labels, ids, segs)
print(f"applied {len(batch)} strips, {n} samples")
