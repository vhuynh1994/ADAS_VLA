#!/usr/bin/env bash
# Round 5: Qwen2.5-VL-7B-Instruct as the base VLM (the Qwen2.5-VL size Qualcomm AI Hub optimizes for SA8650P /
# SA8775P; Apache-2.0), QLoRA on ds_v3 with the v3 recipe (class share 0.55, lr 2e-4 -> cosine, 3 epochs ~ v1+v2+v3),
# shard-wise merge, eval v5 vs v3 on ds_v3 (val + Nexar crash test), policy sweep, demo (run + explain), HTML reports.
#
#   nohup scripts/pipeline_v5_7b.sh > outputs/v5_7b.log 2>&1 &
#
# Knobs (environment variables):
#   V5_EXTRA     applied to EVERY 7B stage, e.g. "--set vlm.quantize_vision=true" when training does not fit 8 GB
#                (the ViT must be quantized the same way at eval time)
#   TRAIN_EXTRA  training only, e.g. "--optimizer paged_adamw_8bit --lora-r 8"
#   BRAKE_WEIGHT loss weight of braking samples (default 1.0 = the v3 recipe; 2.0 moved v4 along the trade-off curve)
#   SKIP_SMOKE=1 skip the 20-step memory / speed smoke test
#   SAVE_EVERY   resumable checkpoint every N optimizer steps (default 100 ~ 35 min on the 4060); a rerun of this
#                script continues the training from it (and skips the smoke test and a finished training)
#   TAG          model name (default v5), e.g. TAG=v5b V5_EXTRA="" for a second run next to v5
#   NO_INHIBIT=1 do not block suspend: by default the whole run holds a systemd sleep + lid-switch inhibitor, so
#                closing the laptop lid no longer suspends it (a suspend hung the CUDA context on 2026-10-11).
#                Keep the laptop on a desk, on AC, ventilated, while it trains.
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ -z ${NO_INHIBIT:-} && -z ${ADAS_VLA_INHIBITED:-} ]] && command -v systemd-inhibit >/dev/null; then
  exec env ADAS_VLA_INHIBITED=1 systemd-inhibit --what=sleep:handle-lid-switch --who="adas-vla v5 pipeline" \
    --why="VLM fine-tune / eval on the GPU" --mode=block "$0" "$@"
fi
A=".venv/bin/adas-vla"; P=".venv/bin/python"
C="--config configs/vlm_7b.yaml"
D=data/ds_v3/labels.jsonl
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|cap_pixels_per_frame|^\s*$'
DEMO_CLIP="data/raw/qutegocentric--Australian_Roads_Dashcam_Driving/Crash/20260113_Zyo1DICmicY_004.mp4"
EXTRA="${V5_EXTRA:-}"
TAG=${TAG:-v5}         # model name: checkpoints/lora-$TAG, models/adas-vlm-$TAG, outputs/*_$TAG*
OUT=checkpoints/lora-$TAG
TRAIN="--epochs 3 --grad-accum 8 --lr 2e-4 --max-class-share 0.55 --workers 2 --brake-weight ${BRAKE_WEIGHT:-1.0} ${TRAIN_EXTRA:-}"
wait_gpu() {
  local need=$1 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [[ ${free:-0} -ge $need ]] && return
    echo "   [$(date +%T)] waiting for GPU memory: ${free} MiB free, need ${need} MiB"; sleep 300
  done
}
mkdir -p outputs
if [[ ! -f models/Qwen--Qwen2.5-VL-7B-Instruct/config.json ]]; then
  echo "== [$(date +%T)] downloading Qwen/Qwen2.5-VL-7B-Instruct (~16 GB)"
  $P scripts/fetch_hf.py Qwen/Qwen2.5-VL-7B-Instruct || exit 1
fi
trained() { [[ -f $OUT/adas_vla_meta.json && ! -d $OUT/resume ]]; }  # finished runs remove their resume state
if [[ -z ${SKIP_SMOKE:-} && ! -d $OUT/resume ]] && ! trained; then
  wait_gpu 7500
  echo "== [$(date +%T)] smoke: 20 optimizer steps on the 7B (peak VRAM + ETA of the full run)"
  $A train $C --data $D --output checkpoints/lora-$TAG-smoke $TRAIN --max-steps 20 $EXTRA 2>&1 | grep --line-buffered -vE "$F"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "smoke failed (out of memory?): retry with V5_EXTRA='--set vlm.quantize_vision=true' TRAIN_EXTRA='--optimizer paged_adamw_8bit --lora-r 8'"; exit 1; }
fi
if trained; then
  echo "== [$(date +%T)] $OUT already trained: skipping the fine-tune"
else
  wait_gpu 7500
  echo "== [$(date +%T)] fine-tune $TAG (base Qwen2.5-VL-7B-Instruct, ds_v3; resumable every ${SAVE_EVERY:-100} steps)"
  $A train $C --data $D --output $OUT $TRAIN --save-every "${SAVE_EVERY:-100}" --resume $EXTRA 2>&1 \
    | grep --line-buffered -vE "$F"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
fi
echo "== [$(date +%T)] merge $TAG (shard by shard)"
$A merge $C --adapter $OUT --out models/adas-vlm-$TAG 2>&1 | grep --line-buffered -vE "$F"
[[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
wait_gpu 6000
echo "== [$(date +%T)] eval $TAG (val)"
$A eval $C --data $D --split val --set vlm.model_id=models/adas-vlm-$TAG $EXTRA --report outputs/eval_${TAG}_ds3.jsonl \
  2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] eval $TAG (Nexar crash test)"
$A eval $C --data $D --split test_nexar --set vlm.model_id=models/adas-vlm-$TAG $EXTRA \
  --report outputs/eval_nexar_${TAG}_ds3.jsonl 2>&1 | grep --line-buffered -vE "$F"
for split in val test_nexar; do  # v3 reference on the same ds_v3 splits (reused when the v4 pipeline produced it)
  out=outputs/eval_v3_ds3.jsonl; [[ $split == test_nexar ]] && out=outputs/eval_nexar_v3_ds3.jsonl
  if [[ ! -s $out ]]; then
    wait_gpu 5000
    echo "== [$(date +%T)] eval v3 ($split)"
    $A eval --data $D --split $split --set vlm.model_id=models/adas-vlm-v3 --report $out 2>&1 | grep --line-buffered -vE "$F"
  fi
done
echo "== [$(date +%T)] cautious policy sweep ($TAG)"
$P scripts/sweep_policy.py outputs/eval_${TAG}_ds3.jsonl --data $D --nexar outputs/eval_nexar_${TAG}_ds3.jsonl \
  2>&1 | grep --line-buffered -vE "$F"
wait_gpu 6000
echo "== [$(date +%T)] demo $TAG: run (VLM + gate), then explain (LLM alone: the 7B and Qwen3-4B do not fit together)"
$A run $C --source "$DEMO_CLIP" --output outputs/demo_crash_${TAG}.mp4 --log outputs/demo_crash_${TAG}.jsonl --ego-speed 45 \
  --set vlm.model_id=models/adas-vlm-$TAG --set vlm.generate_reason=true $EXTRA 2>&1 | grep --line-buffered -vE "$F"
$A explain --log outputs/demo_crash_${TAG}.jsonl --set llm.language=vi 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] reports"
$A report --run v3=outputs/eval_v3_ds3.jsonl --run $TAG=outputs/eval_${TAG}_ds3.jsonl --data-dir data/ds_v3 \
  --out outputs/report_${TAG}_val.html --title "$TAG (Qwen2.5-VL-7B) vs v3 (3B) - held-out val (ds_v3)" 2>&1 | grep -vE "$F"
$A report --run v3=outputs/eval_nexar_v3_ds3.jsonl --run $TAG=outputs/eval_nexar_${TAG}_ds3.jsonl --data-dir data/ds_v3 \
  --out outputs/report_${TAG}_nexar.html --title "$TAG (Qwen2.5-VL-7B) vs v3 (3B) - Nexar crash test (ds_v3)" 2>&1 | grep -vE "$F"
echo "== [$(date +%T)] $TAG pipeline done"
