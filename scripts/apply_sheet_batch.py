#!/usr/bin/env python3
"""Apply visual-review decisions for contact sheets: {sheet: {number: "exclude" | {field: value}}}."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adas_vla.datasets.curate import apply_sheet_review  # noqa: E402

labels, sheets_dir, batch = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
n = sum(apply_sheet_review(labels, sheets_dir, sheet, {int(k): v for k, v in fixes.items()})
        for sheet, fixes in batch.items())
excluded = sum(v == "exclude" for fixes in batch.values() for v in fixes.values())
print(f"recorded {n} reviews ({excluded} excluded) from {len(batch)} sheets")
