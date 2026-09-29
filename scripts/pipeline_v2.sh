#!/usr/bin/env bash
# Round 2: lighter class balancing (55%), merge LoRA into the weights, evaluate the merged model.
set -euo pipefail
cd "$(dirname "$0")/.."
A=".venv/bin/adas-vla"
D=data/ds_v2/labels.jsonl
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|^\s*$'
echo "== [$(date +%T)] fine-tune v2"
$A train --data $D --output checkpoints/lora-v2 --epochs 2 --grad-accum 8 --lr 2e-4 --max-class-share 0.55 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] merge v2"
$A merge --adapter checkpoints/lora-v2 --out models/adas-vlm-v2 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] eval v2 (merged)"
$A eval --data $D --split val --set vlm.model_id=models/adas-vlm-v2 --report outputs/eval_v2.jsonl 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] done"
