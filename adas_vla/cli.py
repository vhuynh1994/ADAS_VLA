"""Command line interface: `adas-vla <command> --help`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set vlm.language=vi (repeatable)")
    p.add_argument("--ego-speed", type=float, default=None, help="ego speed in km/h (no CAN bus on video)")
    p.add_argument("--no-vlm", action="store_true", help="perception + rules only, do not load the VLM")


def _load_cfg(args):
    from .config import load_config

    cfg = load_config(args.config, args.overrides)
    if args.ego_speed is not None:
        cfg.ego_speed_kmh = args.ego_speed
    if args.no_vlm:
        cfg.vlm.enabled = False
    return cfg


def cmd_run(args) -> None:
    import cv2

    from .events import frame_record
    from .hud import HUD
    from .pipeline import ADASPipeline
    from .sources import iter_frames, source_fps
    from .types import EgoState

    cfg = _load_cfg(args)
    if args.vlm_mode:
        cfg.vlm.mode = args.vlm_mode
    if args.vlm_every:
        cfg.vlm.every_n_frames = args.vlm_every
    pipe = ADASPipeline(cfg)
    hud = HUD()
    writer = None
    log_file = open(args.log, "w") if args.log else None
    ego = EgoState(speed_kmh=cfg.ego_speed_kmh)
    n, t_start = 0, time.perf_counter()
    try:
        for idx, t, frame in iter_frames(args.source, args.max_frames):
            res = pipe.process(frame, idx, t, ego)
            vis = hud.draw(frame, res)
            if args.output:
                if writer is None:
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             source_fps(args.source), (w, h))
                writer.write(vis)
            if log_file:
                log_file.write(json.dumps(frame_record(res), ensure_ascii=False) + "\n")
            if args.show:
                cv2.imshow("ADAS-VLA", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            n += 1
            if idx % 30 == 0:
                print(f"frame {idx:5d}  {res.decision.label:<28} {res.decision.source:<6} "
                      f"target {res.decision.target_speed_kmh:5.1f} km/h  alerts "
                      f"{','.join(a.kind for a in res.alerts) or '-'}", flush=True)
    finally:
        pipe.close()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"\nProcessed {n} frames in {elapsed:.1f}s ({n / max(elapsed, 1e-6):.1f} FPS)")
    if pipe.vlm_calls:
        lat = sorted(pipe.vlm_latencies)
        print(f"VLM calls: {pipe.vlm_calls}, unparsable: {pipe.vlm_failures}, "
              f"latency p50 {lat[len(lat) // 2]:.2f}s  max {lat[-1]:.2f}s")
    if args.output:
        print(f"Annotated video: {args.output}")


def cmd_analyze(args) -> None:
    import cv2

    from .hud import HUD
    from .pipeline import ADASPipeline
    from .types import EgoState

    cfg = _load_cfg(args)
    cfg.vlm.mode = "sync"
    cfg.vlm.every_n_frames = 1
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"Cannot read image: {args.image}")
    pipe = ADASPipeline(cfg)
    res = pipe.process(frame, 0, 0.0, EgoState(speed_kmh=cfg.ego_speed_kmh))
    print("=== Perception ===")
    print(res.context.summary_text())
    if pipe.vlm is not None:
        print("\n=== VLM raw output ===")
        print(pipe.last_raw)
    print("\n=== Final decision (after safety gate) ===")
    print(json.dumps(res.decision.to_dict(), indent=2, ensure_ascii=False))
    for a in res.alerts:
        print(f"ALERT [{a.level}] {a.kind}: {a.message}")
    out = args.output or str(Path("outputs") / (Path(args.image).stem + "_analyzed.jpg"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, HUD().draw(frame, res))
    print(f"\nAnnotated image: {out}")


def cmd_chat(args) -> None:
    from .pipeline import ADASPipeline
    from .reasoning.prompts import alerts_text
    from .types import EgoState

    cfg = _load_cfg(args)
    cfg.vlm.enabled = True
    frame = _read_frame(args.image, args.frame)
    pipe = ADASPipeline(cfg)
    ctx = pipe.perceive(frame, 0, 0.0, EgoState(speed_kmh=cfg.ego_speed_kmh))
    _, alerts = pipe.safety.arbitrate(ctx, None)
    context_text = ctx.summary_text() + alerts_text(alerts)
    print("Perception summary:\n" + context_text)
    print("\nAsk about the scene (/scene to reprint perception, /exit to quit).")
    history: list[tuple[str, str]] = []
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/exit", "/quit", "exit", "quit"):
            break
        if q == "/scene":
            print(context_text)
            continue
        t0 = time.perf_counter()
        answer = pipe.vlm.chat(frame, context_text, cfg.ego_speed_kmh, history, q)
        print(f"copilot> {answer}\n({time.perf_counter() - t0:.1f}s)")
        history.append((q, answer))


def _read_frame(path: str, frame_idx: int = 0):
    import cv2

    from .sources import IMAGE_EXTS

    if Path(path).suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(path)
    else:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        frame = frame if ok else None
    if frame is None:
        sys.exit(f"Cannot read frame {frame_idx} from {path}")
    return frame


def cmd_label(args) -> None:
    from .training.autolabel import autolabel

    cfg = _load_cfg(args)
    autolabel(cfg, args.source, Path(args.out), every=args.every, max_frames=args.max_frames)


def cmd_train(args) -> None:
    from .training.finetune import TrainArgs, train

    cfg = _load_cfg(args)
    train(cfg, TrainArgs(data=args.data, output_dir=args.output, epochs=args.epochs, lr=args.lr,
                         grad_accum=args.grad_accum, lora_r=args.lora_r, val_data=args.val_data,
                         max_class_share=args.max_class_share, init_adapter=args.init_adapter,
                         workers=args.workers, brake_weight=args.brake_weight))



def cmd_eval(args) -> None:
    from .training.evaluate import evaluate

    cfg = _load_cfg(args)
    if args.adapter:
        cfg.vlm.adapter_path = args.adapter
    evaluate(cfg, args.data, limit=args.limit, report_path=args.report, split=args.split,
             reviewed_only=args.reviewed_only)


def cmd_demo(args) -> None:
    """VLM + VLA + LLM together: drive the video, and let the LLM explain each intervention as it happens."""
    import cv2

    from .events import EventDetector, frame_record
    from .hud import HUD
    from .pipeline import ADASPipeline
    from .reasoning.llm import TextLLM
    from .sources import iter_frames, source_fps
    from .types import EgoState

    cfg = _load_cfg(args)
    cfg.vlm.enabled = True
    pipe = ADASPipeline(cfg)  # VLM (vision-language) inside the VLA loop
    llm = TextLLM(cfg.llm)  # text-only LLM for explanations
    hud, detector = HUD(), EventDetector(args.min_gap)
    ego = EgoState(speed_kmh=cfg.ego_speed_kmh)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, copilot, copilot_until = None, None, -1.0
    events = []
    with open(out.with_suffix(".events.jsonl"), "w") as log:
        for idx, t, frame in iter_frames(args.source, args.max_frames):
            res = pipe.process(frame, idx, t, ego)
            new = detector.update(t, [a.kind for a in res.alerts], res.decision.source)
            if new:
                event = {**frame_record(res), "kinds": new}
                text, latency = llm.explain(event)
                copilot, copilot_until = text, t + args.show_s
                events.append({**event, "explanation": text, "llm_latency_s": round(latency, 2)})
                log.write(json.dumps(events[-1], ensure_ascii=False) + "\n")
                print(f"[t={t:5.1f}s] {', '.join(new)} -> {res.decision.label}\n   LLM ({latency:.1f}s): {text}",
                      flush=True)
            vis = hud.draw(frame, res, copilot if t <= copilot_until else None)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), source_fps(args.source), (w, h))
            writer.write(vis)
    pipe.close()
    if writer:
        writer.release()
    lat = sorted(pipe.vlm_latencies)
    print(f"\nVLM calls: {pipe.vlm_calls} (unparsable {pipe.vlm_failures}, p50 {lat[len(lat) // 2] if lat else 0:.2f}s)"
          f" | LLM explanations: {len(events)} | video: {out}")


def cmd_explain(args) -> None:
    from .events import extract_events, load_run_log
    from .reasoning.llm import TextLLM

    cfg = _load_cfg(args)
    events = extract_events(load_run_log(args.log), min_gap_s=args.min_gap)
    print(f"{len(events)} ADAS events in {args.log}")
    if not events:
        return
    llm = TextLLM(cfg.llm)
    for ev in events[:args.max_events]:
        text, latency = llm.explain(ev)
        print(f"\n[t={ev['t']:.1f}s frame {ev['frame']}] {', '.join(ev['kinds'])} -> {ev['decision']['longitudinal']} / {ev['decision']['lateral']}")
        print(f"  {text}  ({latency:.1f}s)")


def cmd_build_dataset(args) -> None:
    cfg = _load_cfg(args)
    out = Path(args.out)
    if args.kind == "nexar":
        from .datasets import nexar

        n = nexar.build(cfg, Path(args.root or "data/raw/nexar-ai--nexar_collision_prediction"), out,
                        max_videos=args.max_frames)
    elif args.kind == "comma2k19":
        from .datasets import comma2k19

        n = comma2k19.build(cfg, Path(args.zip), out, max_segments=args.segments, every_s=args.every_s or 2.0,
                            perception_stride=args.stride, val_percent=args.val_percent,
                            skip_from=[Path(p) for p in args.skip_from])
    else:
        from .datasets import clips

        default_root, factory = clips.PRESETS[args.kind]
        source = factory(Path(args.root or default_root))
        if args.every_s:
            source.every_s = args.every_s
        n = clips.build(cfg, source, out, val_percent=args.val_percent, max_frames_per_video=args.max_frames,
                        stride=args.stride)
    print(f"\nAdded {n} samples to {out / 'labels.jsonl'}")


def cmd_report(args) -> None:
    from .report import build_report

    runs = dict(r.split("=", 1) for r in args.run)
    notes = Path(args.notes).read_text() if args.notes else ""
    build_report(runs, args.data_dir, args.out, args.title, notes)


def cmd_review(args) -> None:
    from .review import serve

    serve(args.data, args.port)


def cmd_split(args) -> None:
    from .datasets.common import resplit

    print(resplit(Path(args.data), args.val_percent, args.source))


def cmd_merge(args) -> None:
    """Fold a LoRA adapter into the base weights (bf16) so inference has no adapter overhead."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from .config import resolve_model

    cfg = _load_cfg(args)
    base = resolve_model(cfg.vlm.model_id)
    print(f"Merging {args.adapter} into {base} (bf16, on CPU)...")
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(base).save_pretrained(args.out)
    print(f"Saved merged model to {args.out}. Use it with: --set vlm.model_id={args.out} "
          f"--set vlm.adapter_path=null")


def cmd_export(args) -> None:
    from .deploy.export import export_detector

    cfg = _load_cfg(args)
    export_detector(cfg, args.out, imgsz=args.imgsz, qnn_arch=args.qnn_arch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adas-vla", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the full pipeline on a video / camera / image folder")
    _add_common(p)
    p.add_argument("--source", required=True, help="video file, camera index (0), RTSP URL or image folder")
    p.add_argument("--output", help="write annotated video (mp4)")
    p.add_argument("--log", help="write per-frame decisions as JSONL")
    p.add_argument("--show", action="store_true", help="display a window (q to quit)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--vlm-mode", choices=["sync", "async"])
    p.add_argument("--vlm-every", type=int, help="run the VLM every N frames")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="analyze one image and print the decision JSON")
    _add_common(p)
    p.add_argument("--image", required=True)
    p.add_argument("--output")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("chat", help="interactive copilot Q&A about an image or video frame")
    _add_common(p)
    p.add_argument("--image", required=True, help="image or video file")
    p.add_argument("--frame", type=int, default=0, help="frame index when --image is a video")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("label", help="auto-label video frames with the teacher VLM + safety gate")
    _add_common(p)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True, help="output dataset folder (frames/ + labels.jsonl)")
    p.add_argument("--every", type=int, default=15, help="label every N-th frame")
    p.add_argument("--max-frames", type=int)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("train", help="LoRA / QLoRA fine-tune the VLM on a labels.jsonl dataset")
    _add_common(p)
    p.add_argument("--data", required=True)
    p.add_argument("--val-data")
    p.add_argument("--output", default="checkpoints/lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--init-adapter", help="warm start from this LoRA adapter directory")
    p.add_argument("--max-class-share", type=float, default=0.35,
                   help="cap each label's share of an epoch (1.0 = no balancing)")
    p.add_argument("--workers", type=int, default=2, help="DataLoader workers preparing samples for the GPU")
    p.add_argument("--brake-weight", type=float, default=1.0,
                   help="loss weight of DECELERATE/BRAKE/STOP samples (e.g. 2.0 to fight under-braking)")
    p.set_defaults(func=cmd_train)


    p = sub.add_parser("eval", help="evaluate action accuracy / JSON validity / latency on a dataset")
    _add_common(p)
    p.add_argument("--data", required=True)
    p.add_argument("--adapter", help="LoRA adapter directory")
    p.add_argument("--limit", type=int)
    p.add_argument("--report", help="write per-sample results as JSONL")
    p.add_argument("--split", help="only records with this split (e.g. val)")
    p.add_argument("--reviewed-only", action="store_true", help="only human-reviewed records")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("demo", help="VLM + VLA + LLM together on a video (LLM explains interventions live)")
    _add_common(p)
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="outputs/demo_all.mp4")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--min-gap", type=float, default=3.0, help="seconds before the same alert is explained again")
    p.add_argument("--show-s", type=float, default=4.0, help="seconds an explanation stays on screen")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("explain", help="LLM explains ADAS interventions found in a run log")
    _add_common(p)
    p.add_argument("--log", required=True, help="JSONL written by `adas-vla run --log`")
    p.add_argument("--max-events", type=int, default=10)
    p.add_argument("--min-gap", type=float, default=2.0, help="merge repeats of an alert within N seconds")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("build-dataset", help="build labels.jsonl from public dashcam data")
    _add_common(p)
    p.add_argument("kind", choices=["comma2k19", "nexar", "australian", "uk", "udacity"])
    p.add_argument("--out", default="data/ds_v1")
    p.add_argument("--zip", default="data/raw/commaai--comma2k19/raw_data/Chunk_1.zip", help="comma2k19 chunk")
    p.add_argument("--segments", type=int, default=60, help="comma2k19: number of 1-minute segments")
    p.add_argument("--skip-from", action="append", default=[], metavar="LABELS.jsonl",
                   help="comma2k19: skip segments already in these datasets and their val routes (extra data = train)")
    p.add_argument("--stride", type=int, default=4, help="run perception every N frames (sample frames are always processed)")
    p.add_argument("--root", help="clip folder (defaults per preset)")
    p.add_argument("--every-s", type=float, help="seconds between samples")
    p.add_argument("--val-percent", type=int, default=20)
    p.add_argument("--max-frames", type=int, help="clips: max frames read per video")
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("report", help="HTML report comparing eval runs (from `eval --report`)")
    p.add_argument("--run", action="append", required=True, metavar="NAME=REPORT.jsonl")
    p.add_argument("--data-dir", required=True, help="dataset folder (to show frames)")
    p.add_argument("--out", default="outputs/report.html")
    p.add_argument("--title", default="ADAS-VLA evaluation")
    p.add_argument("--notes", help="text file with notes to include")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("review", help="web page to review/correct labels (keyboard driven)")
    p.add_argument("--data", required=True, help="labels.jsonl")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("split", help="re-split train/val by whole videos to hit a target val share")
    p.add_argument("--data", required=True)
    p.add_argument("--val-percent", type=int, default=20)
    p.add_argument("--source", help="only re-split records from this source")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("merge", help="merge a LoRA adapter into the base VLM weights")
    _add_common(p)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True, help="output directory, e.g. models/adas-vlm-v1")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("export", help="export the detector for Qualcomm QAIRT/QNN (static-shape ONNX)")
    _add_common(p)
    p.add_argument("--out", default="outputs/deploy")
    p.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    p.add_argument("--qnn-arch", help="also run Ultralytics' QNN export for this HTP arch (e.g. 79, 81)")
    p.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
