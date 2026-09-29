#!/usr/bin/env bash
# Round 4: 2-frame VLM (current frame + the frame 0.5 s earlier) fine-tuned from v3 on ds_v3, then compared with
# v3 on the same refreshed dataset (val + Nexar crash test), policy sweep, demo video and HTML reports.
set -uo pipefail
cd "$(dirname "$0")/.."
A=".venv/bin/adas-vla"
D=data/ds_v3/labels.jsonl
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|cap_pixels_per_frame|^\s*$'
TWO="--set vlm.prev_frame_s=0.5"
DEMO_CLIP="data/raw/qutegocentric--Australian_Roads_Dashcam_Driving/Crash/20260113_Zyo1DICmicY_004.mp4"
until grep -q "data stage done" outputs/v4_data.log 2>/dev/null; do sleep 30; done
echo "== [$(date +%T)] fine-tune v4 (2-frame input, base models/adas-vlm-v3)"
$A train --data $D --output checkpoints/lora-v4 --epochs 1 --grad-accum 8 --lr 1e-4 --max-class-share 0.55 \
  --workers 2 --brake-weight 2.0 $TWO 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] merge v4"
$A merge --adapter checkpoints/lora-v4 --out models/adas-vlm-v4 2>&1 | grep --line-buffered -vE "$F"
for m in v4 v3; do
  extra=""; [[ $m == v4 ]] && extra=$TWO
  echo "== [$(date +%T)] eval $m (val)"
  $A eval --data $D --split val --set vlm.model_id=models/adas-vlm-$m $extra --report outputs/eval_${m}_ds3.jsonl \
    2>&1 | grep --line-buffered -vE "$F"
  echo "== [$(date +%T)] eval $m (Nexar crash test)"
  $A eval --data $D --split test_nexar --set vlm.model_id=models/adas-vlm-$m $extra \
    --report outputs/eval_nexar_${m}_ds3.jsonl 2>&1 | grep --line-buffered -vE "$F"
done
echo "== [$(date +%T)] cautious policy sweep (v4)"
.venv/bin/python scripts/sweep_policy.py outputs/eval_v4_ds3.jsonl --data $D --nexar outputs/eval_nexar_v4_ds3.jsonl \
  2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] demo v4"
$A demo --source "$DEMO_CLIP" --output outputs/demo_crash_v4.mp4 --ego-speed 45 --set vlm.model_id=models/adas-vlm-v4 \
  $TWO --set vlm.generate_reason=true --set llm.language=vi 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] reports"
$A report --run v3=outputs/eval_v3_ds3.jsonl --run v4=outputs/eval_v4_ds3.jsonl --data-dir data/ds_v3 \
  --out outputs/report_v4_val.html --title "v4 (2-frame) vs v3 - held-out val (ds_v3)" 2>&1 | grep -vE "$F"
$A report --run v3=outputs/eval_nexar_v3_ds3.jsonl --run v4=outputs/eval_nexar_v4_ds3.jsonl --data-dir data/ds_v3 \
  --out outputs/report_v4_nexar.html --title "v4 (2-frame) vs v3 - Nexar crash test (ds_v3)" 2>&1 | grep -vE "$F"
echo "== [$(date +%T)] v4 pipeline done"
