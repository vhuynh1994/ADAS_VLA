#!/usr/bin/env bash
# Pre-download model weights into models/ (resumable, sha256-verified; safe to re-run).
#   ./scripts/download_models.sh            # smoke-test models + main models
#   ./scripts/download_models.sh smoke      # only the small models used for a quick test
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE=(HuggingFaceTB/SmolVLM-256M-Instruct Qwen/Qwen2.5-0.5B-Instruct)
MAIN=(Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen3-4B-Instruct-2507)
models=("${SMOKE[@]}")
[[ "${1:-all}" == "smoke" ]] || models+=("${MAIN[@]}")
mkdir -p "$ROOT/models/yolop"
[[ -s "$ROOT/models/yolop/yolop-640-640.onnx" ]] || curl -fL --retry 10 -C - -o "$ROOT/models/yolop/yolop-640-640.onnx" \
  https://github.com/hustvl/YOLOP/raw/main/weights/yolop-640-640.onnx   # lane + drivable-area CNN (MIT)
for m in "${models[@]}"; do
  "$ROOT/.venv/bin/python" "$ROOT/scripts/fetch_hf.py" "$m" || echo "!! $m incomplete - re-run this script to resume"
done
