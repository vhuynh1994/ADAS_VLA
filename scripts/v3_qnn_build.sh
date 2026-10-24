#!/usr/bin/env bash
# QAIRT 2.46 build of adas-vlm-v3 for the HTP (inputs from scripts/v3_qnn.py export / calib / overrides):
#   ONNX -> qairt-converter (+ encoding overrides) -> qairt-quantizer (calibration list) -> context binaries
#   bash scripts/v3_qnn_build.sh convert   NAME...   # NAME = vision_encoder | ar128_cl1024_1_of_4 | ar1_cl1024_1_of_4 ...
#   bash scripts/v3_qnn_build.sh quantize  NAME...   # W4A16 LLM (LM head 8-bit), W8A16 ViT, KV 8-bit, bias 32-bit
#   bash scripts/v3_qnn_build.sh ctx SOC             # SOC = sa8650 (HTP v73, soc 52) | sa8797 (HTP v81, soc 72)
#   bash scripts/v3_qnn_build.sh info SOC            # graph sizes, spill-fill, VTCM from the context binaries
# Files live in data/qnn_v3 (SSD); the docker wrapper mounts that folder's parent (WORK).
# Graph names come from the float DLC file names (ar128_cl1024_1_of_4 ...), which is what Genie expects.
# Memory: one tool run at a time (the quantizer on a 9-layer part peaks at ~10 GB); docker gets MEM (default 20g).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
Q=$(readlink -f "$HERE/data")/qnn_v3
QAIRT=${QAIRT:-$HOME/workspaces/qairt_env/qairt.sh}
export WORK=${WORK:-$(dirname "$Q")}
export DOCKER_MEM=${MEM:-20g}
PARTS=4
CL=1024

quant_flags() {  # $1 = graph name
  case "$1" in
    vision_encoder) echo "--act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 --use_per_channel_quantization" ;;
    *) echo "--act_bitwidth 16 --weights_bitwidth ${WBITS:-4} --bias_bitwidth 32 --use_per_channel_quantization" ;;
  esac
}

case "${1:-}" in
  convert)
    shift
    for n in "$@"; do
      ov=()
      [ -f "$Q/overrides/$n.json" ] && ov=(--quantization_overrides "$Q/overrides/$n.json")
      echo "== convert $n ${ov[*]:-}"
      (cd "$Q/$n" && $QAIRT qairt-converter --input_network "$n.onnx" --output_path "$n.dlc" "${ov[@]}" \
        > convert.log 2>&1) || { tail -30 "$Q/$n/convert.log"; exit 1; }
      ls -la "$Q/$n/$n.dlc"
    done ;;
  quantize)
    shift
    for n in "$@"; do
      list=$Q/calib/$n/input_list.txt
      echo "== quantize $n ($(wc -l < "$list") samples): $(quant_flags "$n") ${QFLAGS:-}"
      # shellcheck disable=SC2046
      (cd "$Q/$n" && $QAIRT qairt-quantizer --input_dlc "$n.dlc" --output_dlc "$n.q.dlc" \
        --input_list "$list" $(quant_flags "$n") ${QFLAGS:-} > quantize.log 2>&1) \
        || { tail -30 "$Q/$n/quantize.log"; exit 1; }
      ls -la "$Q/$n/$n.q.dlc"
    done ;;
  ctx)
    soc=${2:?sa8650 or sa8797}
    case "$soc" in
      sa8650) dev='{"soc_id": 52, "dsp_arch": "v73"}' ;;
      sa8797) dev='{"soc_id": 72, "dsp_arch": "v81"}' ;;
      *) echo "unknown SOC $soc"; exit 2 ;;
    esac
    out=$Q/bin/$soc
    mkdir -p "$out"
    names=(vision_encoder)
    for i in $(seq 1 $PARTS); do names+=("ar128_cl${CL}_${i}_of_${PARTS}" "ar1_cl${CL}_${i}_of_${PARTS}"); done
    graphs=$(printf '"%s",' "${names[@]}"); graphs=${graphs%,}
    cat > "$out/htp_ext.json" <<EOF
{"graphs": [{"graph_names": [$graphs], "vtcm_mb": 8, "O": 3}], "devices": [$dev],
 "context": {"weight_sharing_enabled": true}, "memory": {"mem_type": "shared_buffer"}}
EOF
    cat > "$out/htp_config.json" <<EOF
{"backend_extensions": {"shared_library_path": "libQnnHtpNetRunExtensions.so", "config_file_path": "$out/htp_ext.json"}}
EOF
    build() {  # $1 = binary name, rest = graph names (weights shared between them)
      local bin=$1; shift
      local dlcs=""
      for n in "$@"; do dlcs+="$Q/$n/$n.q.dlc,"; done
      echo "== $soc: $bin <- $*"
      (cd "$out" && $QAIRT qnn-context-binary-generator --backend libQnnHtp.so \
        --model libQnnModelDlc.so --dlc_path "${dlcs%,}" --binary_file "$bin" --output_dir "$out" \
        --config_file "$out/htp_config.json" > "$out/$bin.log" 2>&1) || { tail -30 "$out/$bin.log"; exit 1; }
      ls -la "$out/$bin.bin"
    }
    if [ -n "${ONLY:-}" ]; then targets=("$ONLY"); else targets=(vision_encoder $(seq 1 $PARTS)); fi
    for t in "${targets[@]}"; do
      if [ "$t" = vision_encoder ]; then build vision_encoder vision_encoder
      else build "part${t}_of_${PARTS}" "ar128_cl${CL}_${t}_of_${PARTS}" "ar1_cl${CL}_${t}_of_${PARTS}"; fi
    done ;;
  info)
    soc=${2:?sa8650 or sa8797}
    for b in "$Q/bin/$soc"/*.bin; do
      $QAIRT qnn-context-binary-utility --context_binary "$b" --json_file "${b%.bin}.json" > /dev/null 2>&1 || true
    done
    python3 - "$Q/bin/$soc" <<'EOF'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    if f.endswith(("htp_ext.json", "htp_config.json")):
        continue
    info = json.load(open(f))["info"]
    print(f"{os.path.basename(f)[:-5]}: {os.path.getsize(f[:-5] + '.bin') / 2**20:.0f} MiB, soc {info.get('socModel')}")
    for g in info["graphs"]:
        gi = g["info"]
        b1 = gi.get("graphBlobInfo", {}).get("info", {})
        b2 = gi.get("graphBlobInfoV2", {})
        print(f"  {gi['graphName']}: spill-fill {b1.get('spillFillBufferSize', 0) / 2**20:.1f} MiB, vtcm {b1.get('vtcmSize')} MB,"
              f" O{b1.get('optimizationLevel')}, shared weights {b2.get('sharedWeightsSize', 0) / 2**20:.0f} MiB,"
              f" io {b2.get('ioTensorSize', 0) / 2**20:.1f} MiB")
EOF
    ;;
  *) sed -n 2,11p "$0"; exit 2 ;;
esac
