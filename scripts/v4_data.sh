#!/usr/bin/env bash
# Round 4 data stage: rebuild exactly the ds_v2 samples with the current perception into data/ds_v3 (adds the
# frame 0.5 s earlier + the structured lead object; refreshed context text). Labels and splits are carried over
# through the ds_v2 overlays, which are keyed by sample id / group. UK and Udacity samples are all excluded.
# Three builds run in parallel (video decoding is CPU-bound) into separate folders, then are merged.
set -uo pipefail
cd "$(dirname "$0")/.."
A=".venv/bin/adas-vla"
REF=data/ds_v2/labels.jsonl
OUT=data/ds_v3
PARTS=data/ds_v3_parts
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|^\s*$'
comma() {  # $1 = chunk, $2 = output folder
  $A build-dataset comma2k19 --out "$2" --zip data/raw/commaai--comma2k19/raw_data/$1.zip --only-from $REF \
    --set vlm.enabled=false 2>&1 | grep --line-buffered -vE "$F" | sed -u "s/^/[$1] /"
}
clips() {
  for kind in australian nexar; do
    $A build-dataset $kind --out $PARTS/clips --set vlm.enabled=false 2>&1 | grep --line-buffered -vE "$F" \
      | sed -u "s/^/[$kind] /"
  done
}
echo "== [$(date +%T)] building comma2k19 Chunk_1 / Chunk_2 and the clips in parallel"
comma Chunk_1 $OUT & comma Chunk_2 $PARTS/c2 & clips &
wait
echo "== [$(date +%T)] merging"
for part in $PARTS/c2 $PARTS/clips; do
  cat $part/labels.jsonl >> $OUT/labels.jsonl
  find $part/frames -type f -exec mv -t $OUT/frames {} +
done
cp data/ds_v2/labels.reviews.jsonl data/ds_v2/labels.splits.json $OUT/
.venv/bin/python - <<'PY'
from collections import Counter
from adas_vla.training.data import load_records
old = {r["id"]: r for r in load_records("data/ds_v2/labels.jsonl") if r["source"] in ("comma2k19", "australian", "nexar")}
new = {r["id"]: r for r in load_records("data/ds_v3/labels.jsonl")}
missing, extra = set(old) - set(new), set(new) - set(old)
both = set(old) & set(new)
print(f"ds_v2 samples {len(old)}, ds_v3 {len(new)}: missing {len(missing)}, extra {len(extra)}, "
      f"same split {sum(old[i]['split'] == new[i]['split'] for i in both)}, "
      f"same longitudinal label {sum(old[i]['target']['longitudinal'] == new[i]['target']['longitudinal'] for i in both)}")
print("ds_v3 splits:", Counter(r["split"] for r in new.values()))
print("with previous frame:", sum(bool(r.get("image_prev")) for r in new.values()), "| with lead:",
      sum(r.get("lead") is not None for r in new.values()))
PY
echo "== [$(date +%T)] data stage done"
