#!/usr/bin/env bash
# Deploy / start / stop vla_stream_server on the SA8650P board from the PC.
#   bash board/board_server.sh deploy      # copy the built server + models + QNN libs to $DIR on the board
#   bash board/board_server.sh start       # runs until stop (log: $DIR/server.log); keeps the HTP in burst mode
#   bash board/board_server.sh stop | log
# BUNDLE = local folder with lib/ (qnn libs for aarch64-qnx800) and models/ (context binaries); DET / LANE = model files.
set -euo pipefail
BOARD=${BOARD:-root@192.168.0.73}
DIR=${DIR:-/data/adas_vla_perf}
PORT=${PORT:-50052}
DET=${DET:-models/yolo11s_w8a8.bin}
LANE=${LANE:-models/yolop_lane_w8a16_o3v8.bin}
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 $BOARD"
SSHF="ssh -f -o BatchMode=yes -o ConnectTimeout=8 $BOARD"  # -f: the QNX sshd keeps the session open while the server runs
# QNX shell: no ps/kill in PATH -> find the pid via /proc/<pid>/exefile, kill with the ksh builtin.
STOP='T=/usr/bin/toybox; for p in /proc/[0-9]*; do case "$($T cat $p/exefile 2>/dev/null)" in *vla_stream_server*) kill ${p#/proc/}; echo "stopped ${p#/proc/}";; esac; done; $T sleep 1; for p in /proc/[0-9]*; do case "$($T cat $p/exefile 2>/dev/null)" in *vla_stream_server*) kill -9 ${p#/proc/};; esac; done; true'

case "${1:-}" in
  deploy)
    $SSH "$STOP" > /dev/null
    $SSH "/usr/bin/toybox mkdir -p $DIR/bin"
    scp -q -o BatchMode=yes "$HERE/build/vla_stream_server" "$BOARD:$DIR/bin/"
    echo "copied to $BOARD:$DIR/bin/vla_stream_server (models and lib/ must already be in $DIR)" ;;
  start)
    $SSH "$STOP" > /dev/null
    $SSHF "cd $DIR && /usr/bin/toybox chmod 755 bin/vla_stream_server && export LD_LIBRARY_PATH=\$(pwd)/lib \
      CDSP_LIBRARY_PATH='/mnt/etc/images/dsp;/mnt/dsplib/image/dsp/cdsp0' ADSP_LIBRARY_PATH=\$(pwd)/lib && \
      /usr/bin/toybox nohup bin/vla_stream_server --det $DET --lane $LANE --port $PORT --backend lib/libQnnHtp.so \
      --system lib/libQnnSystem.so --perf burst < /dev/null > server.log 2>&1 &" < /dev/null > /dev/null 2>&1
    for _ in $(seq 1 40); do
      if $SSH "/usr/bin/toybox grep -q -E 'listening|failed|cannot|unexpected|expected' $DIR/server.log" 2>/dev/null; then break; fi
      sleep 0.5
    done
    $SSH "/usr/bin/toybox grep -v -E 'Alloc2 Support|^\$' $DIR/server.log" ;;
  stop) $SSH "$STOP" ;;
  log) $SSH "/usr/bin/toybox grep -v -E 'Alloc2 Support|^\$' $DIR/server.log" ;;
  *) sed -n 2,6p "$0"; exit 2 ;;
esac
