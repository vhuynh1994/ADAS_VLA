#!/usr/bin/env bash
# Build board/genie_htp_shim.c for the SA8650P (QNX 8.0, aarch64le) and install it on the board:
#   QNN_SDK_ROOT=<QAIRT 2.46 root> bash board/genie_htp_shim.sh [root@192.168.0.73]
# On the board: /data/adas_vla_genie/shim/libQnnHtp.so (the shim) + libQnnHtpReal.so (copy of /mnt/qnn2.46/libQnnHtp.so).
# Run Genie with LD_LIBRARY_PATH=/data/adas_vla_genie/shim:/mnt/qnn2.46 and LD_PRELOAD=/mnt/lib64/libfastrpc_pmem.so
# (QNX keeps rpcmem_alloc/free/to_fd there, Genie looks for them in libcdsprpc.so).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/build
BOARD=${1:-root@192.168.0.73}
QNX_SDP=${QNX_SDP:-$HOME/qnx800}
QNN_SDK_ROOT=${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the QAIRT SDK root (include/QNN)}
# shellcheck disable=SC1091
source "$QNX_SDP/qnxsdp-env.sh" > /dev/null
mkdir -p "$OUT"
qcc -Vgcc_ntoaarch64le -O2 -Wall -shared -fPIC -I"$QNN_SDK_ROOT/include/QNN" \
  -Wl,-soname,libQnnHtpShim.so "$HERE/genie_htp_shim.c" -o "$OUT/libQnnHtp.so"
file "$OUT/libQnnHtp.so"
ssh -o BatchMode=yes "$BOARD" '/usr/bin/toybox mkdir -p /data/adas_vla_genie/shim &&
  /usr/bin/toybox cp /mnt/qnn2.46/libQnnHtp.so /data/adas_vla_genie/shim/libQnnHtpReal.so'
scp -q -o BatchMode=yes "$OUT/libQnnHtp.so" "$BOARD:/data/adas_vla_genie/shim/libQnnHtp.so"
echo "installed in $BOARD:/data/adas_vla_genie/shim"
