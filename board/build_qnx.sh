#!/usr/bin/env bash
# Cross-compile vla_stream_server for the SA8650P board (QNX 8.0, aarch64le) with the QNX SDP on the PC.
# Usage: bash board/build_qnx.sh [out_dir]   (QNX_SDP, QNN_SDK_ROOT can be overridden)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/build}
QNX_SDP=${QNX_SDP:-$HOME/qnx800}
QNN_SDK_ROOT=${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the QAIRT SDK root (include/QNN)}
# shellcheck disable=SC1091
source "$QNX_SDP/qnxsdp-env.sh" > /dev/null
mkdir -p "$OUT"
qcc -Vgcc_ntoaarch64le_cxx -std=gnu++17 -O2 -Wall -I"$QNN_SDK_ROOT/include/QNN" -I"$HERE" \
  "$HERE/vla_stream_server.cpp" -o "$OUT/vla_stream_server" -lsocket
file "$OUT/vla_stream_server"
