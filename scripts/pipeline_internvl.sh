#!/usr/bin/env bash
# InternVL3.5-2B (the VLM that runs on the SA8650P board) fine-tuned on ds_v3 with the v3 recipe (class share 0.55,
# lr 2e-4 -> cosine, 3 epochs, QLoRA), shard-wise merge, eval vs v3 on ds_v3 (val + Nexar crash test), policy sweep,
# demo, HTML reports. Same mechanics as pipeline_v5_7b.sh:
#
#   STOP_AT=07:30 nohup scripts/pipeline_internvl.sh >> outputs/internvl.log 2>&1 &
#
#   TAG          model name (default ivl1): checkpoints/lora-$TAG, models/adas-vlm-$TAG, outputs/*_$TAG*
#   TRAIN_EXTRA  extra training flags; BRAKE_WEIGHT (default 1.0); SKIP_SMOKE=1; SAVE_EVERY (default 100)
#   STOP_AT      HH:MM: the training saves a checkpoint and stops (overnight only); rerun the same command to continue
#   NO_INHIBIT=1 do not hold the sleep / lid-switch inhibitor
# Needs models/OpenGVLab--InternVL3_5-2B-HF (scripts/fetch_hf.py) and the SSD with data/ mounted.
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ -z ${NO_INHIBIT:-} && -z ${ADAS_VLA_INHIBITED:-} ]] && command -v systemd-inhibit >/dev/null; then
  exec env ADAS_VLA_INHIBITED=1 systemd-inhibit --what=sleep:handle-lid-switch --who="adas-vla InternVL pipeline" \
    --why="VLM fine-tune / eval on the GPU" --mode=block "$0" "$@"
fi
A=".venv/bin/adas-vla"; P=".venv/bin/python"
C="--config configs/vlm_internvl.yaml"
D=data/ds_v3/labels.jsonl
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|^\s*$'
DEMO_CLIP="data/raw/qutegocentric--Australian_Roads_Dashcam_Driving/Crash/20260113_Zyo1DICmicY_004.mp4"
TAG=${TAG:-ivl1}
OUT=checkpoints/lora-$TAG
TRAIN="--epochs 3 --grad-accum 8 --lr 2e-4 --max-class-share 0.55 --workers 2 --brake-weight ${BRAKE_WEIGHT:-1.0} ${TRAIN_EXTRA:-}"
[[ -f $D ]] || { echo "$D not found: is the SSD mounted?"; exit 1; }
[[ -f models/OpenGVLab--InternVL3_5-2B-HF/config.json ]] || { echo "run scripts/fetch_hf.py OpenGVLab/InternVL3_5-2B-HF"; exit 1; }
trained() { [[ -f $OUT/adas_vla_meta.json && ! -d $OUT/resume ]]; }
if [[ -z ${SKIP_SMOKE:-} && ! -d $OUT/resume ]] && ! trained; then
  echo "== [$(date +%T)] smoke: 20 optimizer steps (peak VRAM + ETA of the full run)"
  $A train $C --data $D --output $OUT-smoke $TRAIN --max-steps 20 2>&1 | grep --line-buffered -vE "$F"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "smoke failed"; exit 1; }
fi
if trained; then
  echo "== [$(date +%T)] $OUT already trained: skipping the fine-tune"
else
  echo "== [$(date +%T)] fine-tune $TAG (base InternVL3.5-2B, ds_v3; resumable every ${SAVE_EVERY:-100} steps)"
  $A train $C --data $D --output $OUT $TRAIN --save-every "${SAVE_EVERY:-100}" --resume ${STOP_AT:+--stop-at $STOP_AT} \
    2>&1 | grep --line-buffered -vE "$F"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
  trained || { echo "== [$(date +%T)] training paused (STOP_AT / SIGTERM): rerun this script to continue"; exit 0; }
fi
echo "== [$(date +%T)] merge $TAG (shard by shard)"
$A merge $C --adapter $OUT --out models/adas-vlm-$TAG 2>&1 | grep --line-buffered -vE "$F"
[[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
for split in val test_nexar; do
  out=outputs/eval_${TAG}_ds3.jsonl; [[ $split == test_nexar ]] && out=outputs/eval_nexar_${TAG}_ds3.jsonl
  echo "== [$(date +%T)] eval $TAG ($split)"
  $A eval $C --data $D --split $split --set vlm.model_id=models/adas-vlm-$TAG --report $out 2>&1 \
    | grep --line-buffered -vE "$F"
done
echo "== [$(date +%T)] cautious policy sweep ($TAG)"
$P scripts/sweep_policy.py outputs/eval_${TAG}_ds3.jsonl --data $D --nexar outputs/eval_nexar_${TAG}_ds3.jsonl \
  2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] demo $TAG (VLM + gate on the AU crash clip)"
$A run $C --source "$DEMO_CLIP" --output outputs/demo_crash_${TAG}.mp4 --log outputs/demo_crash_${TAG}.jsonl \
  --ego-speed 45 --set vlm.model_id=models/adas-vlm-$TAG 2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] reports"
for s in val nexar; do
  ref=outputs/eval_v3_ds3.jsonl; new=outputs/eval_${TAG}_ds3.jsonl
  [[ $s == nexar ]] && { ref=outputs/eval_nexar_v3_ds3.jsonl; new=outputs/eval_nexar_${TAG}_ds3.jsonl; }
  $A report --run v3=$ref --run $TAG=$new --data-dir data/ds_v3 --out outputs/report_${TAG}_$s.html \
    --title "$TAG (InternVL3.5-2B) vs v3 (Qwen2.5-VL-3B) - $s (ds_v3)" 2>&1 | grep -vE "$F"
done
echo "== [$(date +%T)] $TAG pipeline done"
