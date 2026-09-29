#!/usr/bin/env bash
# Download public sample dashcam clips (Udacity self-driving car nanodegree, MIT license) and a street image.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/samples"
mkdir -p "$DIR"
fetch() {
  local url="$1" out="$DIR/$2"
  if [[ -s "$out" ]]; then echo "exists: $out"; return; fi
  echo "downloading $2 ..."
  curl -fL --retry 5 --retry-delay 3 -C - -o "$out" "$url"
}
fetch https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4 highway_short.mp4
fetch https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/project_video.mp4 highway_traffic.mp4
fetch https://ultralytics.com/images/bus.jpg street.jpg
ls -lh "$DIR"
