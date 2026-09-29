#!/usr/bin/env bash
# Round 3 (overnight): wait for the data stage, warm-start fine-tune from v2 on the enlarged train set, merge,
# evaluate on the fixed val split and on the held-out Nexar crash test split, then render before/after demos.
set -uo pipefail
cd "$(dirname "$0")/.."
A=".venv/bin/adas-vla"
D=data/ds_v2/labels.jsonl
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|^\s*$'
DEMO_CLIP="data/raw/qutegocentric--Australian_Roads_Dashcam_Driving/Crash/20260113_Zyo1DICmicY_004.mp4"
until grep -q "data stage done" outputs/v3_data.log 2>/dev/null; do sleep 30; done
echo "== [$(date +%T)] fine-tune v3 (warm start from v2)"
$A train --data $D --output checkpoints/lora-v3 --epochs 1 --grad-accum 8 --lr 1e-4 --max-class-share 0.55 \
  --init-adapter checkpoints/lora-v2 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] merge v3"
$A merge --adapter checkpoints/lora-v3 --out models/adas-vlm-v3 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] eval v3 (val)"
$A eval --data $D --split val --set vlm.model_id=models/adas-vlm-v3 --report outputs/eval_v3.jsonl 2>&1 | grep --line-buffered -vE "$F"
for m in base v2 v3; do
  echo "== [$(date +%T)] eval $m (Nexar crash test)"
  model="Qwen/Qwen2.5-VL-3B-Instruct"; [[ $m != base ]] && model="models/adas-vlm-$m"
  $A eval --data $D --split test_nexar --set vlm.model_id=$model --report outputs/eval_nexar_$m.jsonl 2>&1 | grep --line-buffered -vE "$F"
done
for m in base v3; do
  echo "== [$(date +%T)] demo $m"
  model="Qwen/Qwen2.5-VL-3B-Instruct"; [[ $m != base ]] && model="models/adas-vlm-$m"
  $A demo --source "$DEMO_CLIP" --output outputs/demo_crash_$m.mp4 --ego-speed 45 --set vlm.model_id=$model \
    --set vlm.generate_reason=true --set llm.language=vi 2>&1 | grep --line-buffered -vE "$F"
done
echo "== [$(date +%T)] v3 pipeline done"
