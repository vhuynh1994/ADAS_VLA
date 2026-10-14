#!/usr/bin/env python3
"""Benchmark the on-board VLM server (board/vlm_server.sh start) with the project's real decision prompt.

Sends the same decision requests (frames of a video + a fixed perception summary) and reports the server timings
(image + ViT, first token, total, tokens) and the latency seen on the PC, so server configs can be compared:

  GEN=greedy-text-generator-nsp1.json bash board/vlm_server.sh start
  python scripts/bench_board_vlm.py --video data/samples/highway_traffic.mp4 --label baseline
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2

from adas_vla.config import VLMConfig
from adas_vla.reasoning.board_vlm import BoardVLM

CONTEXT = """Lane: ego lane detected, left and right boundaries visible.
Detected objects (monocular distance estimates):
- car, ahead, ~18 m, IN EGO LANE, closing at 2.1 m/s, TTC 8.6 s
- truck, front-right, ~31 m
- car, front-left, ~24 m
- car, front-right, ~45 m
- motorcycle, front-left, ~12 m, cutting in
- car, ahead, ~60 m"""


def frames(video: str, times: list[float]):
    cap = cv2.VideoCapture(video)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            yield t, frame
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="data/samples/highway_traffic.mp4")
    ap.add_argument("--times", type=float, nargs="+", default=[2, 6, 10, 14, 18, 22, 26, 30])
    ap.add_argument("--repeat", type=int, default=2, help="passes over the frames (the first request is a warm-up)")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="outputs/bench_board_vlm.jsonl")
    args = ap.parse_args()

    cfg = VLMConfig(backend="board", thinking=args.thinking)
    vlm = BoardVLM(cfg)
    shots = list(frames(args.video, args.times))
    rows = []
    for rep in range(args.repeat):
        for t, frame in shots:
            decision, raw, latency = vlm.decide(frame, CONTEXT, 62.0, 80.0)
            row = {"label": args.label, "t": t, "pass": rep, "latency_s": round(latency, 3), **vlm.last_timing,
                   "parsed": decision is not None, "raw": raw}
            if decision is not None:
                row["action"] = f"{decision.longitudinal.value}/{decision.lateral.value}"
            rows.append(row)
            print(f"{t:5.1f}s pass {rep}: {latency:5.2f} s | image {row['image_ms']:4.0f} ms, first token "
                  f"{row['first_token_ms']:5.0f} ms, total {row['total_ms']:5.0f} ms, {row['tokens']:3d} tok | {raw[:90]}")
    rows = rows[1:]  # drop the warm-up request
    med = lambda k: statistics.median(r[k] for r in rows)  # noqa: E731
    print(f"\n[{args.label}] {len(rows)} requests, median: PC latency {med('latency_s'):.2f} s, image+ViT "
          f"{med('image_ms'):.0f} ms, first token {med('first_token_ms'):.0f} ms, total {med('total_ms'):.0f} ms, "
          f"{med('tokens'):.0f} tokens; parsed {sum(r['parsed'] for r in rows)}/{len(rows)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
