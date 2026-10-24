#!/usr/bin/env bash
# Round 7: richer VLM output. A teacher (Qwen2.5-VL-7B, 4-bit) explains the KNOWN decision of each train frame
# (scene / hazards / reason, checked against the perception list by scripts/explain_labels.py), then v3 is
# fine-tuned to answer the decision FIRST (same fields, the action is still the first generated token) and the
# explanation after it (`--set vlm.explain=true`; loss = mean over the decision tokens + mean over the explanation
# tokens). Then v6 vs v3 on ds_v3 (val + Nexar crash test, decision-only decoding), the explanations of the held-out
# val frames vs the teacher's, policy sweep, demo video, HTML reports.
#
#   scripts/pipeline_v6.sh [labels|train|all] > outputs/v6.log 2>&1      # default all; rerun to continue
#
# Every stage is resumable and the whole run stops at STOP_AT (overnight only): a rerun of the same command the next
# night continues where it stopped (labels ~6 h for ~8,300 frames at ~2.6 s, training ~4 h, evals ~1 h).
#   labels  teacher explanations: VAL_EXPLAIN val frames (default 300, for the explanation check), then the train
#           selection (every non-KEEP frame + KEEP frames, default 4400: the class balance of the v3 recipe)
#   train   gate (labels complete, acceptance >= MIN_ACCEPT, default 0.85), dataset data/ds_v3x, fine-tune from
#           models/adas-vlm-v3 (1 epoch, lr 1e-4), merge to models/adas-vlm-$TAG, evals, reports, demo
# Knobs: STOP_AT HH:MM (default 07:30; empty = no stop), TAG (default v6), KEEP, VAL_EXPLAIN, MIN_ACCEPT,
# BRAKE_WEIGHT (default 1.0), NO_INHIBIT=1 (by default the run blocks suspend + lid switch: a suspend hung the CUDA
# context on 2026-10-11; keep the laptop on a desk, on AC, ventilated).
set -uo pipefail
cd "$(dirname "$0")/.."
stage=${1:-all}
[[ $stage == labels || $stage == train || $stage == all ]] || { echo "usage: $0 [labels|train|all]"; exit 2; }
if [[ -z ${NO_INHIBIT:-} && -z ${ADAS_VLA_INHIBITED:-} ]] && command -v systemd-inhibit >/dev/null; then
  exec env ADAS_VLA_INHIBITED=1 systemd-inhibit --what=sleep:handle-lid-switch --who="adas-vla v6 pipeline" \
    --why="teacher labels / VLM fine-tune on the GPU" --mode=block "$0" "$@"
fi
A=".venv/bin/adas-vla"; P=".venv/bin/python"
D=data/ds_v3/labels.jsonl
X=data/ds_v3x
F='UserWarning|warn\(|x\[seq\]|pos_axes_slices|Loading weights|it/s\]|cap_pixels_per_frame|^\s*$'
DEMO_CLIP="data/raw/qutegocentric--Australian_Roads_Dashcam_Driving/Crash/20260113_Zyo1DICmicY_004.mp4"
TAG=${TAG:-v6}         # checkpoints/lora-$TAG, models/adas-vlm-$TAG, outputs/*_$TAG*
OUT=checkpoints/lora-$TAG
STOP_AT=${STOP_AT-07:30}
KEEP=${KEEP:-4400}; VAL_EXPLAIN=${VAL_EXPLAIN:-300}
EXPLAIN="--set vlm.explain=true"
wait_gpu() {
  local need=$1 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [[ ${free:-0} -ge $need ]] && return
    echo "   [$(date +%T)] waiting for GPU memory: ${free} MiB free, need ${need} MiB"; sleep 300
  done
}
# past the night's STOP_AT (and before the evening): do not start another stage
past_stop() {
  [[ -n $STOP_AT ]] || return 1
  local now stop
  now=$((10#$(date +%H%M))); stop=$((10#${STOP_AT/:/}))
  if (( now >= stop && now < 1800 )); then echo "== [$(date +%T)] past STOP_AT $STOP_AT: rerun this command tonight"; return 0; fi
  return 1
}
[[ -r $D ]] || { echo "$D is not readable: is the data SSD mounted?"; exit 1; }
mkdir -p outputs

if [[ $stage == labels || $stage == all ]]; then
  for split in val train; do
    past_stop && exit 0
    lim="--limit $VAL_EXPLAIN"; [[ $split == train ]] && lim="--keep $KEEP"
    wait_gpu 7000
    echo "== [$(date +%T)] teacher explanations ($split)"
    $P scripts/explain_labels.py generate --split $split $lim ${STOP_AT:+--stop-at $STOP_AT} 2>&1 | grep --line-buffered -vE "$F"
    rc=${PIPESTATUS[0]}
    [[ $rc -eq 2 ]] && { echo "== [$(date +%T)] labels paused at STOP_AT: rerun this command tonight"; exit 0; }
    [[ $rc -eq 0 ]] || exit 1
  done
fi
[[ $stage == labels ]] && { $P scripts/explain_labels.py status --val "$VAL_EXPLAIN" --keep "$KEEP"; exit 0; }

trained() { [[ -f $OUT/adas_vla_meta.json && ! -d $OUT/resume ]]; }  # finished runs remove their resume state
if [[ ! -f models/adas-vlm-$TAG/config.json ]]; then
  if ! trained; then
    if [[ ! -d $OUT/resume ]]; then  # a fresh run: the gate, then the dataset from the accepted explanations
      echo "== [$(date +%T)] teacher explanations: status"
      $P scripts/explain_labels.py status --val "$VAL_EXPLAIN" --keep "$KEEP" --min-accept "${MIN_ACCEPT:-0.85}"
      rc=$?
      [[ $rc -eq 1 ]] && { echo "labels incomplete: run '$0 labels' first"; exit 1; }
      [[ $rc -eq 3 ]] && { echo "acceptance too low: look at the rejections (explain_labels.py show / status) before training"; exit 1; }
      $P scripts/explain_labels.py merge --out $X || exit 1
    fi
    past_stop && exit 0
    wait_gpu 7000
    echo "== [$(date +%T)] fine-tune $TAG (base models/adas-vlm-v3, $X, explanation after the decision; resumable every 100 steps)"
    $A train --data $X/labels.jsonl --val-data $X/val_explained.jsonl --output $OUT --epochs 1 --grad-accum 8 \
      --lr 1e-4 --max-class-share 0.55 --workers 2 --brake-weight "${BRAKE_WEIGHT:-1.0}" --save-every 100 --resume \
      ${STOP_AT:+--stop-at $STOP_AT} $EXPLAIN 2>&1 | grep --line-buffered -vE "$F"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
    trained || { echo "== [$(date +%T)] training paused (STOP_AT / SIGTERM): rerun this command tonight"; exit 0; }
  fi
  echo "== [$(date +%T)] merge $TAG"
  $A merge --adapter $OUT --out models/adas-vlm-$TAG 2>&1 | grep --line-buffered -vE "$F"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
fi
# evals, each skipped once its report exists (written under a temporary name until it is complete)
evaluate() {  # name report args...
  local name=$1 report=$2; shift 2
  [[ -s $report ]] && return 0
  past_stop && exit 0
  wait_gpu 5000
  echo "== [$(date +%T)] eval $name"
  $A eval "$@" --report "$report.part" 2>&1 | grep --line-buffered -vE "$F"
  [[ ${PIPESTATUS[0]} -eq 0 ]] && mv "$report.part" "$report"
}
M="--set vlm.model_id=models/adas-vlm-$TAG $EXPLAIN"
evaluate "$TAG (val, decision only)" outputs/eval_${TAG}_ds3.jsonl --data $D --split val $M
evaluate "$TAG (Nexar crash test, decision only)" outputs/eval_nexar_${TAG}_ds3.jsonl --data $D --split test_nexar $M
evaluate "$TAG (val frames the teacher explained, full answer)" outputs/explain_${TAG}_val.jsonl \
  --data $X/val_explained.jsonl $M --set vlm.generate_reason=true --set vlm.max_new_tokens=160
evaluate "v3 (the same frames, its own answer with reason)" outputs/explain_v3_val.jsonl \
  --data $X/val_explained.jsonl --set vlm.generate_reason=true
for split in val test_nexar; do  # v3 reference on the same ds_v3 splits (from round 4)
  out=outputs/eval_v3_ds3.jsonl; [[ $split == test_nexar ]] && out=outputs/eval_nexar_v3_ds3.jsonl
  evaluate "v3 ($split)" $out --data $D --split $split
done
echo "== [$(date +%T)] explanations: $TAG and v3 vs the teacher (held-out val frames)"
for m in $TAG v3; do
  $P scripts/explain_labels.py score --report outputs/explain_${m}_val.jsonl --data $X/val_explained.jsonl \
    --examples 8 > outputs/explain_${m}_val.txt
done
cat outputs/explain_${TAG}_val.txt; head -8 outputs/explain_v3_val.txt
echo "== [$(date +%T)] cautious policy sweep ($TAG)"
$P scripts/sweep_policy.py outputs/eval_${TAG}_ds3.jsonl --data $D --nexar outputs/eval_nexar_${TAG}_ds3.jsonl \
  2>&1 | grep --line-buffered -vE "$F"
echo "== [$(date +%T)] reports"
$A report --run v3=outputs/eval_v3_ds3.jsonl --run $TAG=outputs/eval_${TAG}_ds3.jsonl --data-dir data/ds_v3 \
  --out outputs/report_${TAG}_val.html --title "$TAG (explains) vs v3 - held-out val (ds_v3)" 2>&1 | grep -vE "$F"
$A report --run v3=outputs/eval_nexar_v3_ds3.jsonl --run $TAG=outputs/eval_nexar_${TAG}_ds3.jsonl --data-dir data/ds_v3 \
  --out outputs/report_${TAG}_nexar.html --title "$TAG (explains) vs v3 - Nexar crash test (ds_v3)" 2>&1 | grep -vE "$F"
if [[ ! -s outputs/demo_crash_${TAG}.mp4 ]] && ! past_stop; then
  wait_gpu 6000
  echo "== [$(date +%T)] demo $TAG (scene / hazards / reason on the HUD)"
  $A run --source "$DEMO_CLIP" --output outputs/demo_crash_${TAG}.mp4 --log outputs/demo_crash_${TAG}.jsonl \
    --ego-speed 45 --set vlm.model_id=models/adas-vlm-$TAG $EXPLAIN --set vlm.generate_reason=true \
    --set vlm.max_new_tokens=160 2>&1 | grep --line-buffered -vE "$F"
fi
echo "== [$(date +%T)] $TAG pipeline done"
