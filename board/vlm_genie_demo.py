#!/usr/bin/env python3
"""InternVL3.5-2B (Qualcomm AI Hub Genie bundle, W4A16) on the SA8650P: dashcam frames in, answers + timings out.

The PC picks frames from a video, resizes them to the bundle's 448x448 input (NCHW float32, RGB, ImageNet
normalisation), copies them to the board, runs one genie-app script there (the model is loaded once for all frames:
vision encoder on NSP1, LLM on NSP0) and shows each frame with the answer and its timings, then writes a video.

  python board/vlm_genie_demo.py --video data/raw/.../Crash/20260113_Zyo1DICmicY_004.mp4 --times 0 5 9.3 16 24.8 27.6

Board setup (board/README.md, "VLM trên board"): bundle in /data/adas_vla_genie/internvl3_5_2b, genie-app in
/data/adas_vla_genie/bin, board/genie_htp_shim.sh installed; `--setup` once writes the QNX configs (ViT on NSP1).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

import cv2
import numpy as np

BOARD = "root@192.168.0.73"
BUNDLE = "/data/adas_vla_genie/internvl3_5_2b"
SSH = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=15"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img>"
QUESTION = ("You are the camera assistant of a driver-assistance system. In two short sentences: what are the road "
            "situation and the hazards ahead, and should the driver KEEP speed, DECELERATE, BRAKE or STOP?")
ENV = ("ulimit -c 0; export LD_LIBRARY_PATH=/data/adas_vla_genie/shim:/mnt/qnn2.46:$LD_LIBRARY_PATH "
       "CDSP_LIBRARY_PATH='/mnt/etc/images/dsp;/mnt/dsplib/image/dsp/cdsp0' "
       "CDSP1_LIBRARY_PATH='/mnt/etc/images/dsp;/mnt/dsplib/image/dsp/cdsp1' "
       "ADSP_LIBRARY_PATH='/mnt/etc/images/dsp;/mnt/dsplib/image/dsp/cdsp0' "
       "LD_PRELOAD=/mnt/lib64/libfastrpc_pmem.so")


def setup_board() -> None:
    """Derive the QNX configs from the bundle's own: ViT on NSP1 (device_id 1) without vision-param."""
    def cat(name: str) -> dict:
        return json.loads(subprocess.run(["ssh", *SSH, BOARD, f"/usr/bin/toybox cat {BUNDLE}/{name}"],
                                         capture_output=True, text=True, check=True).stdout)

    ext = cat("htp_backend_ext_config.json")
    ext["devices"][0]["device_id"] = 1
    enc = cat("img-enc-htp.json")
    enc["image-encoder"]["engine"]["model"].pop("vision-param", None)
    enc["image-encoder"]["engine"]["backend"]["extensions"] = "htp_ext_nsp1.json"
    for name, cfg in (("htp_ext_nsp1.json", ext), ("img-enc-nsp1.json", enc)):
        subprocess.run(["ssh", *SSH, BOARD, f"/usr/bin/toybox cat > {BUNDLE}/{name}"], input=json.dumps(cfg, indent=2),
                       text=True, check=True)
    print(f"wrote {BUNDLE}/htp_ext_nsp1.json and img-enc-nsp1.json")


def pixel_values(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (448, 448), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()  # NCHW


def genie_script(n: int) -> str:
    head = ["version", "profile create prof", "pipeline config create pipelineConfig",
            "pipeline config bind profile pipelineConfig prof", "pipeline create GeniePipeline pipelineConfig",
            "node config create imageEncoderConfig img-enc-nsp1.json",
            "node config bind profile imageEncoderConfig prof", "node create imageEncoder imageEncoderConfig",
            "node config create lutEncoderConfig text-encoder.json", "node create lutEncoder lutEncoderConfig",
            "node config create textGeneratorConfig text-generator.json",
            "node config bind profile textGeneratorConfig prof", "node create textGenerator textGeneratorConfig",
            "node set textCallback textGenerator GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT",
            "pipeline add GeniePipeline imageEncoder", "pipeline add GeniePipeline lutEncoder",
            "pipeline add GeniePipeline textGenerator",
            "pipeline connect GeniePipeline imageEncoder GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT textGenerator "
            "GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
            "pipeline connect GeniePipeline lutEncoder GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT textGenerator "
            "GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT"]
    body = []
    for i in range(n):
        body += ([] if i == 0 else ["pipeline reset GeniePipeline"]) + [
            "node set textFile lutEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT demo/prefix.txt",
            f"node set image imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT demo/frame_{i}.raw",
            "node set textFile lutEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT demo/suffix.txt",
            "pipeline execute GeniePipeline"]
    tail = ["profile save prof demo/profile.json", "node free imageEncoder", "node free lutEncoder",
            "node free textGenerator", "pipeline free GeniePipeline", "profile free prof"]
    return "\n".join(head + body + tail) + "\n"


def run_on_board(frames: list[np.ndarray], question: str) -> tuple[list[str], list[dict], float]:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "demo"
        d.mkdir()
        for i, f in enumerate(frames):
            pixel_values(f).tofile(d / f"frame_{i}.raw")
        (d / "prefix.txt").write_text(PREFIX)
        (d / "suffix.txt").write_text(f"</img>{question}<|im_end|>\n<|im_start|>assistant\n")
        (d / "run.txt").write_text(genie_script(len(frames)))
        subprocess.run(["ssh", *SSH, BOARD, f"/usr/bin/toybox rm -rf {BUNDLE}/demo"], check=True)
        subprocess.run(["scp", "-q", *SSH, "-r", str(d), f"{BOARD}:{BUNDLE}/"], check=True)
    out = subprocess.run(["ssh", *SSH, BOARD, f"{ENV}; cd {BUNDLE} && ../bin/genie-app -s demo/run.txt 2>&1"],
                         capture_output=True, text=True, timeout=1800).stdout
    answers = [a.strip() for a in re.findall(r"\[BEGIN\]:(.*?)\[END\]", out, re.S)]
    prof = json.loads(subprocess.run(["ssh", *SSH, BOARD, f"/usr/bin/toybox cat {BUNDLE}/demo/profile.json"],
                                     capture_output=True, text=True, check=True).stdout)
    events = {c["name"]: c["events"] for c in prof["components"]}
    runs = [e for e in events.get("pipeline0", []) if e["type"] == "GeniePipeline_execute"]
    vit = [e["duration"] / 1e6 for e in events.get("node0", []) if e["type"] == "GenieNode_setData"]
    init = sum(e.get("init-time", {}).get("value", 0) for c in events.values() for e in c) / 1e6
    stats = [{"vit_s": vit[i] if i < len(vit) else None, "ttft_s": r["time-to-first-token"]["value"] / 1e6,
              "prompt_tok_s": r["prompt-processing-rate"]["value"], "gen_tok_s": r["token-generation-rate"]["value"],
              "tokens": r["num-generated-tokens"]["value"], "total_s": r["duration"] / 1e6} for i, r in enumerate(runs)]
    if len(answers) != len(frames):
        print(out[-3000:])
    return answers, stats, init


def render(frame: np.ndarray, answer: str, st: dict, t: float, width: int = 1280) -> np.ndarray:
    img = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
    lines = textwrap.wrap(answer, 95)[:6]
    head = (f"InternVL3.5-2B W4A16 on SA8650P HTP | t={t:.1f}s | ViT {st['vit_s']:.2f}s (NSP1) | "
            f"first token {st['ttft_s']:.2f}s | {st['gen_tok_s']:.1f} tok/s (NSP0)")
    box_h = 40 + 30 * len(lines)
    over = img.copy()
    cv2.rectangle(over, (0, img.shape[0] - box_h), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    img = cv2.addWeighted(over, 0.65, img, 0.35, 0)
    y = img.shape[0] - box_h + 28
    cv2.putText(img, head, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for ln in lines:
        y += 30
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--times", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--out", default="outputs/vlm_board_demo")
    ap.add_argument("--show", action="store_true", help="show each frame in a window (any key: next)")
    ap.add_argument("--setup", action="store_true", help="first write the QNX configs next to the bundle on the board")
    args = ap.parse_args()
    if args.setup:
        setup_board()
    cap = cv2.VideoCapture(args.video)
    frames = []
    for t in args.times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            raise SystemExit(f"cannot read {args.video} at {t} s")
        frames.append(f)
    answers, stats, init = run_on_board(frames, args.question)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"model load on the board (once): {init:.1f} s")
    shown = []
    for t, f, a, st in zip(args.times, frames, answers, stats):
        print(f"\n[t={t:.1f}s] ViT {st['vit_s']:.2f}s, first token {st['ttft_s']:.2f}s, "
              f"{st['gen_tok_s']:.1f} tok/s, {st['tokens']} tokens, {st['total_s']:.2f}s\n  {a}")
        shown.append(render(f, a, st, t))
    (out / "results.json").write_text(json.dumps([{"t": t, "answer": a, **st} for t, a, st in
                                                  zip(args.times, answers, stats)], indent=2, ensure_ascii=False))
    h, w = shown[0].shape[:2]
    vw = cv2.VideoWriter(str(out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1, (w, h))
    for img in shown:
        for _ in range(4):  # 4 s per frame
            vw.write(img)
        cv2.imwrite(str(out / f"frame_{len(list(out.glob('frame_*.jpg')))}.jpg"), img)
    vw.release()
    print(f"\nsaved {out}/demo.mp4, results.json, frame_*.jpg")
    if args.show:
        for img in shown:
            cv2.imshow("VLM on SA8650P", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
