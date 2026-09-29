#!/usr/bin/env bash
# Baseline eval -> QLoRA fine-tune -> eval of the fine-tuned adapter, on data/ds_v2 (val = whole held-out clips/routes).
set -euo pipefail
cd "$(dirname "$0")/.."
A=".venv/bin/adas-vla"
D=data/ds_v2/labels.jsonl
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|^\s*$'
echo "== [$(date +%T)] baseline eval (zero-shot)"
$A eval --data $D --split val --report outputs/eval_base.jsonl 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] fine-tune v1"
$A train --data $D --output checkpoints/lora-v1 --epochs 2 --grad-accum 8 --lr 2e-4 --max-class-share 0.35 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] eval v1"
$A eval --data $D --split val --adapter checkpoints/lora-v1 --report outputs/eval_v1.jsonl 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] done"
