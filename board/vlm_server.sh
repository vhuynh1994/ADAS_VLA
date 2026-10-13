#!/usr/bin/env bash
# Build / deploy / start / stop the on-board VLM server (vlm_genie_server.cpp: InternVL3.5-2B through Genie).
#   QNN_SDK_ROOT=<QAIRT 2.46> bash board/vlm_server.sh build    # cross-compile with the QNX SDP
#   bash board/vlm_server.sh deploy   # copy the server, write the board configs (vision encoder NSP0, LLM NSP1, greedy)
#   bash board/vlm_server.sh start | stop | log
# Needs the InternVL bundle in $DIR and the libQnnHtp.so shim (genie_htp_shim.sh); see board/README.md.
# The perception server (board_server.sh) shares NSP0 with the vision encoder; the LLM has NSP1 to itself.
set -euo pipefail
BOARD=${BOARD:-root@192.168.0.73}
ROOT=${ROOT:-/data/adas_vla_genie}
DIR=${DIR:-$ROOT/internvl3_5_2b}
PORT=${PORT:-50053}
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 $BOARD"
SSHF="ssh -f -o BatchMode=yes -o ConnectTimeout=8 $BOARD"
STOP='T=/usr/bin/toybox; for p in /proc/[0-9]*; do case "$($T cat $p/exefile 2>/dev/null)" in *vlm_genie_server*) kill ${p#/proc/}; echo "stopped ${p#/proc/}";; esac; done; $T sleep 1; for p in /proc/[0-9]*; do case "$($T cat $p/exefile 2>/dev/null)" in *vlm_genie_server*) kill -9 ${p#/proc/};; esac; done; true'

case "${1:-}" in
  build)
    QNX_SDP=${QNX_SDP:-$HOME/qnx800}
    : "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the QAIRT SDK root}"
    # shellcheck disable=SC1091
    source "$QNX_SDP/qnxsdp-env.sh" > /dev/null
    mkdir -p "$HERE/build"
    qcc -Vgcc_ntoaarch64le_cxx -std=gnu++17 -O2 -Wall -I"$QNN_SDK_ROOT/include/Genie" "$HERE/vlm_genie_server.cpp" \
      -o "$HERE/build/vlm_genie_server" -L"$QNN_SDK_ROOT/lib-safe/aarch64-qnx800" -lGenie -lsocket \
      -Wl,--allow-shlib-undefined  # libGenie.so needs the board's newer libc++, resolved at run time
    file "$HERE/build/vlm_genie_server" ;;
  deploy)
    $SSH "$STOP" > /dev/null
    scp -q -o BatchMode=yes "$HERE/build/vlm_genie_server" "$BOARD:$ROOT/bin/"
    # vision encoder: the bundle's config without vision-param on NSP0; LLM: greedy sampling on NSP1
    BOARD=$BOARD DIR=$DIR python3 - <<'EOF'
import json, os, subprocess
board, d = os.environ["BOARD"], os.environ["DIR"]
ssh = ["ssh", "-o", "BatchMode=yes", board]
cat = lambda n: json.loads(subprocess.run(ssh + [f"/usr/bin/toybox cat {d}/{n}"], capture_output=True, text=True, check=True).stdout)
put = lambda n, c: subprocess.run(ssh + [f"/usr/bin/toybox cat > {d}/{n}"], input=json.dumps(c, indent=2), text=True, check=True)
ext1 = cat("htp_backend_ext_config.json"); ext1["devices"][0]["device_id"] = 1
enc = cat("img-enc-htp.json"); enc["image-encoder"]["engine"]["model"].pop("vision-param", None)
gen = cat("text-generator.json"); gen["text-generator"]["sampler"].update({"top-k": 1, "temp": 0.0001})
gen["text-generator"]["engine"]["backend"]["extensions"] = "htp_ext_nsp1.json"
for name, cfg in (("htp_ext_nsp1.json", ext1), ("img-enc-nsp0.json", enc), ("greedy-text-generator-nsp1.json", gen)):
    put(name, cfg)
print("configs written in", d)
EOF
    ;;
  start)
    $SSH "$STOP" > /dev/null
    $SSHF "ulimit -c 0; cd $ROOT && /usr/bin/toybox chmod 755 bin/vlm_genie_server && \
      export LD_LIBRARY_PATH=$ROOT/shim:/mnt/qnn2.46 LD_PRELOAD=/mnt/lib64/libfastrpc_pmem.so \
      CDSP_LIBRARY_PATH='/mnt/etc/images/dsp;/mnt/dsplib/image/dsp/cdsp0' \
      CDSP1_LIBRARY_PATH='/mnt/etc/images/dsp;/mnt/dsplib/image/dsp/cdsp1' \
      ADSP_LIBRARY_PATH='/mnt/etc/images/dsp;/mnt/dsplib/image/dsp/cdsp0' && \
      /usr/bin/toybox nohup bin/vlm_genie_server --port $PORT --dir $DIR < /dev/null > $ROOT/vlm_server.log 2>&1 &" \
      < /dev/null > /dev/null 2>&1
    for _ in $(seq 1 60); do
      if $SSH "/usr/bin/toybox grep -q -E 'listening|failed|cannot' $ROOT/vlm_server.log" 2>/dev/null; then break; fi
      sleep 1
    done
    $SSH "/usr/bin/toybox grep -v -E 'Alloc2 Support|affinity|^\$' $ROOT/vlm_server.log | /usr/bin/toybox tail -5" ;;
  stop) $SSH "$STOP" ;;
  log) $SSH "/usr/bin/toybox grep -v -E 'Alloc2 Support|affinity|^\$' $ROOT/vlm_server.log | /usr/bin/toybox tail -20" ;;
  *) sed -n 2,7p "$0"; exit 2 ;;
esac
