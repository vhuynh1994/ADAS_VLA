#!/usr/bin/env python3
"""(Re)generate per-clip filmstrips for segment-level labeling; index.json maps strip -> ordered sample ids."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adas_vla.datasets.curate import make_filmstrip  # noqa: E402
from adas_vla.training.data import load_records  # noqa: E402

labels = Path(sys.argv[1])
source = sys.argv[2]
out = Path(sys.argv[3])
meta = Path("data/raw/qutegocentric--Australian_Roads_Dashcam_Driving/video_metadata.csv")
index_path = out / "index.json"
index = json.loads(index_path.read_text()) if index_path.exists() else {}
videos = sorted({r["video"] for r in load_records(labels, include_excluded=True) if r.get("source") == source})
made = 0
for v in videos:
    name = f"{Path(v).parent.name}_{Path(v).stem}".replace(" ", "_") + ".jpg"
    if name in index:
        continue
    index[name] = {"video": v, "ids": make_filmstrip(labels, v, out / name, metadata_csv=meta)}
    made += 1
index_path.write_text(json.dumps(index, indent=1))
print(f"{made} new filmstrips, {len(index)} total in {out}")
