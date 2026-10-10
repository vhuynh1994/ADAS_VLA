#!/usr/bin/env bash
# Live window of the board perception on a video, started as a named user unit (the VS Code snap terminal cannot open
# GUI windows directly, and a systemd-run process does not get Ctrl+C).
#   bash board/demo.sh <video> [extra adas-vla run args]   # e.g. --set vlm.enabled=true --vlm-mode process
#   bash board/demo.sh stop                                # or press q / Esc in the window, or close it
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UNIT=adas-vla-demo
if [ "${1:-}" = stop ]; then systemctl --user stop "$UNIT" 2>/dev/null && echo "stopped $UNIT" || echo "$UNIT not running"; exit 0; fi
[ $# -ge 1 ] || { sed -n 2,5p "$0"; exit 2; }
VIDEO=$(realpath "$1"); shift
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --collect --unit="$UNIT" -p WorkingDirectory="$ROOT" "$ROOT/.venv/bin/adas-vla" run --no-vlm \
  --set perception.backend=board --source "$VIDEO" --show "$@"
echo "stop: bash board/demo.sh stop   (log: journalctl --user -u $UNIT -f)"
