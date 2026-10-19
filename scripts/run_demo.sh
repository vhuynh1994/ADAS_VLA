#!/usr/bin/env bash
# Live ADAS-VLA demo window on a video (q / Esc / closing the window quits; or `scripts/run_demo.sh stop`).
#
#   scripts/run_demo.sh <mode> <video> [ego_speed_kmh]     (default ego speed 60)
#   scripts/run_demo.sh stop
#
# Modes (PC = this laptop's GPU, board = SA8650P at 192.168.0.73):
#   v3          PC: perception + VLM v3 (Qwen2.5-VL-3B fine-tuned, ~1 s) + LLM Qwen3-4B explaining interventions (vi)
#   7b          PC: perception + VLM Qwen2.5-VL-7B (base, zero-shot) with its reason on the HUD; no LLM (8 GB VRAM)
#   7b-think    PC: as 7b, the VLM reasons step by step first (<think>, ~7 s per decision), shown on the HUD
#   board       board: detector + lanes + VLM InternVL3.5-2B (base, action only, ~1.9 s per decision);
#               PC: tracking, distance/TTC, safety gate, HUD
#   board-reason board: as board, the VLM also writes its reason for the HUD (~2.3 s per decision)
#   board-think board: as board, InternVL's thinking mode (~20 s per decision)
#   perception  board: detector + lanes only, rule-based decisions (no VLM)
#
# Results: outputs/demo_<mode>_<video name>.mp4 and .jsonl (per-frame decisions; thinking text included).
# BEV=1 scripts/run_demo.sh <mode> <video> adds a bird's-eye view panel (lanes + objects on the road plane).
# The board modes start the board servers and leave them running: `scripts/run_demo.sh stop` stops everything.
# Examples:
#   scripts/run_demo.sh 7b-think data/samples/highway_traffic.mp4
#   scripts/run_demo.sh v3 data/raw/nexar-ai--nexar_collision_prediction/train/positive/00003.mp4 45
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UNIT=adas-vla-demo
cd "$ROOT"

stop_all() {
  systemctl --user stop "$UNIT" 2>/dev/null && echo "stopped the demo window" || true
  if [ "${1:-}" = board ]; then
    bash board/vlm_server.sh stop || true
    bash board/board_server.sh stop || true
  fi
}

case "${1:-}" in
  stop) stop_all board; exit 0 ;;
  v3|7b|7b-think|board|board-reason|board-think|perception) ;;
  *) sed -n 2,22p "$0"; exit 2 ;;
esac
MODE=$1
[ $# -ge 2 ] || { echo "usage: $0 $MODE <video> [ego_speed_kmh]"; exit 2; }
VIDEO=$(realpath "$2")
EGO=${3:-60}
[ -f "$VIDEO" ] || { echo "video not found: $VIDEO (is the SSD mounted?)"; exit 1; }
NAME=$(basename "${VIDEO%.*}")
OUT="outputs/demo_${MODE}_${NAME}"
A="$ROOT/.venv/bin/adas-vla"
COMMON=(--source "$VIDEO" --ego-speed "$EGO" --output "$OUT.mp4")
[ -n "${BEV:-}" ] && COMMON+=(--bev)  # BEV=1: bird's-eye view panel beside the HUD

case $MODE in
  v3)          CMD=(demo --show --set vlm.mode=process "${COMMON[@]}") ;;
  7b)          CMD=(run --config configs/vlm_7b.yaml --show --vlm-mode process --set vlm.quantize_vision=true
                    --set vlm.generate_reason=true --log "$OUT.jsonl" "${COMMON[@]}") ;;
  7b-think)    CMD=(run --config configs/vlm_7b.yaml --show --vlm-mode process --set vlm.quantize_vision=true
                    --set vlm.thinking=true --log "$OUT.jsonl" "${COMMON[@]}") ;;
  board|board-reason|board-think)
               GEN=fast-text-generator-nsp1.json; THINK=()
               [ "$MODE" = board-reason ] && GEN=greedy-text-generator-nsp1.json
               if [ "$MODE" = board-think ]; then GEN=think-text-generator-nsp1.json; THINK=(--set vlm.thinking=true); fi
               bash board/board_server.sh start | tail -1
               GEN=$GEN bash board/vlm_server.sh start | tail -1
               CMD=(run --show --vlm-mode async --set perception.backend=board --set vlm.backend=board "${THINK[@]}"
                    --log "$OUT.jsonl" "${COMMON[@]}") ;;
  perception)  bash board/board_server.sh start | tail -1
               CMD=(run --show --no-vlm --set perception.backend=board --log "$OUT.jsonl" "${COMMON[@]}") ;;
esac

stop_all
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --collect --unit="$UNIT" -p WorkingDirectory="$ROOT" "$A" "${CMD[@]}" > /dev/null
echo "running: $MODE on $NAME (window opens in a few seconds; the 7B takes ~1 min to load)"
echo "  follow : journalctl --user -u $UNIT -f"
echo "  stop   : scripts/run_demo.sh stop      (or q / Esc / close the window)"
echo "  results: $OUT.mp4 $( [ "$MODE" = v3 ] && echo "$OUT.events.jsonl" || echo "$OUT.jsonl")"
