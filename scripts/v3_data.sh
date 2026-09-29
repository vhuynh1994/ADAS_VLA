#!/usr/bin/env bash
# Round 3 data stage: wait for the extra comma2k19 build and the Nexar download, then build Nexar samples.
set -uo pipefail
cd "$(dirname "$0")/.."
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|^\s*$'
echo "== [$(date +%T)] waiting for the comma2k19 build and the Nexar download"
while pgrep -f "[a]das-vla build-dataset comma2k19" >/dev/null || pgrep -f "[f]etch_hf.py.*nexar" >/dev/null; do sleep 30; done
n=$(ls data/raw/nexar-ai--nexar_collision_prediction/train/positive | grep -c "mp4$")
echo "== [$(date +%T)] Nexar videos on disk: $n/750"
.venv/bin/adas-vla build-dataset nexar --out data/ds_v2 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] data stage done"
