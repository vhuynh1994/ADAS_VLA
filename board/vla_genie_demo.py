#!/usr/bin/env python3
"""VLA on the SA8650P with the default (not fine-tuned) InternVL3.5-2B: the VLM decides, its own LLM explains.

  PC    : dashcam frames -> perception (detector, lanes, distance / TTC) -> context text, safety gate (AEB / FCW)
  board : InternVL3.5-2B (Genie pipeline, ViT on NSP1, LLM on NSP0) <- image + the project's decision prompt
          -> {"longitudinal": ..., "lateral": ..., "target_speed_kmh": ..., "risk": ..., "reason": ...}
  board : the same bundle's LLM (Qwen3-1.7B, text-only Genie dialog: genie_config.json, same part*.bin) <- the ADAS
          event (final action after the gate, alerts, objects) -> explanation for the driver

The AI Hub W4A16 bundle for SA8775P as it is: zero-shot, no fine-tuning, no re-quantization, one model on the board.

  python board/vla_genie_demo.py --video data/raw/.../Crash/20260113_Zyo1DICmicY_004.mp4 --times 0 5 9.3 16.3 24.8 27.6

Board setup: board/README.md ("VLM trên board"); `--setup` once writes the QNX and greedy-decoding configs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlm_genie_demo import BOARD, BUNDLE, ENV, SSH, pixel_values, setup_board  # noqa: E402

LLM_BUNDLE = BUNDLE  # the explanation uses InternVL's own LLM, text only
VLM_HF = "models/OpenGVLab--InternVL3_5-2B-HF"  # processor / chat templates only (no weights loaded)
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"


def ssh(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH, BOARD, cmd], text=True, **kw)


def greedy_configs() -> None:
    """Deterministic decoding on the board (top-k 1) for both models, next to the bundles' own configs."""
    for bundle, name, key in ((BUNDLE, "text-generator.json", "text-generator"), (LLM_BUNDLE, "genie_config.json", "dialog")):
        cfg = json.loads(ssh(f"/usr/bin/toybox cat {bundle}/{name}", capture_output=True, check=True).stdout)
        cfg[key]["sampler"].update({"top-k": 1, "temp": 0.0001})
        ssh(f"/usr/bin/toybox cat > {bundle}/greedy-{name}", input=json.dumps(cfg, indent=2), check=True)


def perceive(video: str, times: list[float], ego_kmh: float):
    """Run perception on every frame up to the last requested time: (contexts of all frames, indices of `times`)."""
    from adas_vla.config import load_config
    from adas_vla.pipeline import ADASPipeline
    from adas_vla.timing import StageTimer
    from adas_vla.types import EgoState

    cfg = load_config()
    pipe = ADASPipeline(cfg, load_vlm=False)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last = int(round(max(times) * fps))
    ctxs, frames = [], {}
    wanted = {int(round(t * fps)): t for t in times}
    for i in range(last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        ctxs.append(pipe.perceive(frame, i, i / fps, EgoState(ego_kmh), StageTimer()))
        if i in wanted:
            frames[i] = frame
    return cfg, ctxs, frames


def vlm_prompts(ctx, ego_kmh: float, cruise_kmh: float, processor) -> tuple[str, str]:
    """The project's decision prompt rendered with InternVL's chat template, split around the image."""
    from PIL import Image

    from adas_vla.reasoning.prompts import DECISION_PREFIX, decision_messages

    msgs = decision_messages(Image.new("RGB", (448, 448)), ctx.summary_text(), ego_kmh, cruise_kmh, "en")
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    i = text.index("<IMG_CONTEXT>")
    return text[:i] + "<img>", "</img>" + text[i + len("<IMG_CONTEXT>"):] + DECISION_PREFIX


def run_vlm(frames: list[np.ndarray], prompts: list[tuple[str, str]]) -> tuple[list[str], list[dict]]:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "vla"
        d.mkdir()
        lines = ["version", "profile create prof", "pipeline config create pc", "pipeline config bind profile pc prof",
                 "pipeline create P pc", "node config create ie img-enc-nsp1.json", "node config bind profile ie prof",
                 "node create img ie", "node config create te text-encoder.json", "node create lut te",
                 "node config create tg greedy-text-generator.json", "node config bind profile tg prof",
                 "node create gen tg", "node set textCallback gen GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT",
                 "pipeline add P img", "pipeline add P lut", "pipeline add P gen",
                 "pipeline connect P img GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT gen GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
                 "pipeline connect P lut GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT gen GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT"]
        for i, (f, (pre, suf)) in enumerate(zip(frames, prompts)):
            pixel_values(f).tofile(d / f"frame_{i}.raw")
            (d / f"pre_{i}.txt").write_text(pre)
            (d / f"suf_{i}.txt").write_text(suf)
            lines += ([] if i == 0 else ["pipeline reset P"]) + [
                f"node set textFile lut GENIE_NODE_TEXT_ENCODER_TEXT_INPUT vla/pre_{i}.txt",
                f"node set image img GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT vla/frame_{i}.raw",
                f"node set textFile lut GENIE_NODE_TEXT_ENCODER_TEXT_INPUT vla/suf_{i}.txt", "pipeline execute P"]
        lines += ["profile save prof vla/profile.json", "node free img", "node free lut", "node free gen",
                  "pipeline free P", "profile free prof"]
        (d / "run.txt").write_text("\n".join(lines) + "\n")
        ssh(f"/usr/bin/toybox rm -rf {BUNDLE}/vla", check=True)
        subprocess.run(["scp", "-q", *SSH, "-r", str(d), f"{BOARD}:{BUNDLE}/"], check=True)
    out = ssh(f"{ENV}; cd {BUNDLE} && ../bin/genie-app -s vla/run.txt 2>&1", capture_output=True, timeout=1800).stdout
    answers = [a.strip() for a in re.findall(r"\[BEGIN\]:(.*?)\[END\]", out, re.S)]
    prof = json.loads(ssh(f"/usr/bin/toybox cat {BUNDLE}/vla/profile.json", capture_output=True, check=True).stdout)
    runs = [e for c in prof["components"] if c["type"] == "pipeline" for e in c["events"]
            if e["type"] == "GeniePipeline_execute"]
    vit = [e["duration"] / 1e6 for c in prof["components"] if c["name"] == "node0" for e in c["events"]
           if e["type"] == "GenieNode_setData"]
    stats = [{"vit_s": vit[i] if i < len(vit) else 0.0, "ttft_s": r["time-to-first-token"]["value"] / 1e6,
              "gen_tok_s": r["token-generation-rate"]["value"], "total_s": r["duration"] / 1e6}
             for i, r in enumerate(runs)]
    if len(answers) != len(frames):
        print(out[-3000:])
    return answers, stats


def run_llm(prompts: list[str]) -> tuple[list[str], list[dict]]:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "explain"
        d.mkdir()
        lines = ["version", "profile create prof", "dialog config create dc greedy-genie_config.json",
                 "dialog config bind profile dc prof", "dialog create D dc"]
        for i, p in enumerate(prompts):
            (d / f"q_{i}.txt").write_text(p)
            lines += ([] if i == 0 else ["dialog reset D"]) + [f"dialog query D explain/q_{i}.txt"]
        lines += ["profile save prof explain/profile.json", "dialog free D", "profile free prof"]
        (d / "run.txt").write_text("\n".join(lines) + "\n")
        ssh(f"/usr/bin/toybox rm -rf {LLM_BUNDLE}/explain", check=True)
        subprocess.run(["scp", "-q", *SSH, "-r", str(d), f"{BOARD}:{LLM_BUNDLE}/"], check=True)
    out = ssh(f"{ENV}; cd {LLM_BUNDLE} && ../bin/genie-app -s explain/run.txt 2>&1", capture_output=True,
              timeout=1800).stdout
    answers = [a.strip() for a in re.findall(r"\[BEGIN\]:(.*?)\[END\]", out, re.S)]
    prof = json.loads(ssh(f"/usr/bin/toybox cat {LLM_BUNDLE}/explain/profile.json", capture_output=True,
                          check=True).stdout)
    runs = [e for c in prof["components"] for e in c["events"] if e["type"] == "GenieDialog_query"]
    stats = [{"ttft_s": r["time-to-first-token"]["value"] / 1e6, "gen_tok_s": r["token-generation-rate"]["value"],
              "total_s": r["duration"] / 1e6} for r in runs]
    if len(answers) != len(prompts):
        print(out[-3000:])
    return answers, stats


def render(frame, t, vlm_raw, vlm_dec, final, alerts, explain, vst, lst, width=1280):
    from PIL import Image, ImageDraw, ImageFont

    img = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
    font, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 18)
    vlm_line = (f"VLM (board, InternVL3.5-2B): {vlm_dec.longitudinal.value} / {vlm_dec.lateral.value}, "
                f"{vlm_dec.target_speed_kmh:.0f} km/h, risk {vlm_dec.risk_level.value}" if vlm_dec
                else f"VLM (board): unparsable: {vlm_raw[:80]}")
    reason = f"  reason: {vlm_dec.reason}" if vlm_dec and vlm_dec.reason else ""
    gate = (f"Safety gate (PC): {final.longitudinal.value} / {final.lateral.value} (source {final.source})"
            + (f"  alerts: {', '.join(alerts)}" if alerts else ""))
    rows = [(vlm_line, (255, 230, 80)), (reason, (230, 230, 230)), (gate, (120, 220, 255))]
    rows += [(ln, (170, 255, 170)) for ln in textwrap.wrap("LLM (board, InternVL's Qwen3-1.7B): " + explain, 110)[:3]]
    rows = [r for r in rows if r[0]]
    timing = (f"t={t:.1f}s | VLM: ViT {vst['vit_s']:.2f}s + first token {vst['ttft_s']:.2f}s, {vst['gen_tok_s']:.0f} tok/s"
              f" | LLM: first token {lst['ttft_s']:.2f}s, {lst['gen_tok_s']:.0f} tok/s")
    box_h = 20 + 30 * len(rows) + 28
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(over).rectangle([0, pil.height - box_h, pil.width, pil.height], fill=(0, 0, 0, 175))
    pil = Image.alpha_composite(pil.convert("RGBA"), over)
    draw = ImageDraw.Draw(pil)
    y = pil.height - box_h + 10
    for text, color in rows:
        draw.text((14, y), text, font=font, fill=color)
        y += 30
    draw.text((14, y), timing, font=small, fill=(200, 200, 200))
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def main() -> None:
    from transformers import AutoProcessor

    from adas_vla.control import SafetySupervisor
    from adas_vla.reasoning.llm import EXPLAIN_SYSTEM_PROMPT, event_prompt
    from adas_vla.reasoning.parser import parse_decision
    from adas_vla.reasoning.prompts import DECISION_PREFIX, LANGUAGE_NAMES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--times", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    ap.add_argument("--ego-speed", type=float, default=45.0)
    ap.add_argument("--language", default="vi")
    ap.add_argument("--out", default="outputs/vla_board_demo")
    ap.add_argument("--setup", action="store_true", help="write the QNX + greedy configs on the board first")
    args = ap.parse_args()
    if args.setup:
        setup_board()
        greedy_configs()

    cfg, ctxs, frames = perceive(args.video, args.times, args.ego_speed)
    idx = sorted(frames)
    processor = AutoProcessor.from_pretrained(VLM_HF)
    cruise = cfg.control.cruise_speed_kmh
    raws, vstats = run_vlm([frames[i] for i in idx], [vlm_prompts(ctxs[i], args.ego_speed, cruise, processor) for i in idx])
    decisions = {}
    for i, raw in zip(idx, raws):
        # zero-shot, the base model sometimes copies the action menu: "KEEP = hold current speed" -> "KEEP"
        d = parse_decision(re.sub(r'"([A-Z_]+) = [^"]*"', r'"\1"', DECISION_PREFIX + raw), args.ego_speed)
        if d is not None:
            d.frame_idx, d.timestamp_s = i, ctxs[i].timestamp_s
        decisions[i] = d

    gate, latest, finals = SafetySupervisor(cfg), None, {}  # replay the gate over every frame, VLM decisions as they come
    for c in ctxs:
        latest = decisions.get(c.frame_idx) or latest
        final, alerts = gate.arbitrate(c, latest)
        if c.frame_idx in decisions:
            finals[c.frame_idx] = (final, alerts)

    tok = processor.tokenizer
    language = LANGUAGE_NAMES.get(args.language, args.language)
    prompts = []
    for i in idx:
        final, alerts = finals[i]
        event = {"t": ctxs[i].timestamp_s, "frame": i, "alerts": [a.__dict__ for a in alerts],
                 "decision": final.to_dict(), "objects": [d.describe() for d in ctxs[i].detections]}
        msgs = [{"role": "system", "content": EXPLAIN_SYSTEM_PROMPT.format(language=language)},
                {"role": "user", "content": event_prompt(event) + f"\n\nAnswer in {language}."}]
        # Qwen3-based LLM: an empty think block selects the non-thinking answer (no "<think>..." in the output)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                       + "<think>\n\n</think>\n\n")
    explains, lstats = run_llm(prompts)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shown, results = [], []
    for k, i in enumerate(idx):
        final, alerts = finals[i]
        kinds = [a.kind for a in alerts]
        dec = decisions[i]
        t = ctxs[i].timestamp_s
        print(f"\n[t={t:.1f}s] VLM: {DECISION_PREFIX}{raws[k][:160]}\n  gate: {final.longitudinal.value} "
              f"({final.source}) {kinds}\n  LLM: {explains[k] if k < len(explains) else ''}")
        shown.append(render(frames[i], t, raws[k], dec, final, kinds, explains[k] if k < len(explains) else "",
                            vstats[k], lstats[k] if k < len(lstats) else {"ttft_s": 0, "gen_tok_s": 0}))
        results.append({"t": t, "context": ctxs[i].summary_text(), "vlm_raw": DECISION_PREFIX + raws[k],
                        "vlm_decision": dec.to_dict() if dec else None, "final": final.to_dict(), "alerts": kinds,
                        "explanation": explains[k] if k < len(explains) else None, "vlm": vstats[k],
                        "llm": lstats[k] if k < len(lstats) else None})
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    h, w = shown[0].shape[:2]
    vw = cv2.VideoWriter(str(out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1, (w, h))
    for k, img in enumerate(shown):
        cv2.imwrite(str(out / f"frame_{k}.jpg"), img)
        for _ in range(6):  # 6 s per frame
            vw.write(img)
    vw.release()
    print(f"\nsaved {out}/demo.mp4, results.json, frame_*.jpg")


if __name__ == "__main__":
    main()
