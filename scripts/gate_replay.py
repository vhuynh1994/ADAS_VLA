#!/usr/bin/env python3
"""Replay the geometry, motion and safety-gate stages offline on recorded detections.

`capture` runs the detector (+ ByteTrack) and the lane detector once on the GPU and stores the raw per-frame
detections (ego-hood boxes included) and lanes. `replay` then re-runs everything downstream of them - hood
filter, monocular distance, closing speed / TTC, cut-in, AEB / FCW / VRU gate - on the CPU in seconds, so
gate and perception parameters can be compared on the same inputs:

  python scripts/gate_replay.py capture --out outputs/gate_replay            # Nexar test videos + AU clips
  python scripts/gate_replay.py replay outputs/gate_replay \\
      --variant main:safety.aeb_hold_s=0,safety.fcw_hold_s=0,safety.hysteresis=1 --variant current:

Depth (perception/depth.py): `capture --set perception.depth.enabled=true` also stores the Depth-Anything statistics
of every frame; `depth DATA` adds them to existing captures (re-reads the videos, does not re-run the detector).
`replay` then compares distance sources on the same frames:

  python scripts/gate_replay.py depth outputs/gate_replay
  python scripts/gate_replay.py replay outputs/gate_replay --variant pinhole:perception.depth.mode=off \\
      --variant depth:perception.depth.mode=replace --variant depth_min:perception.depth.mode=min

Ground truth: Nexar videos have human `time_of_alert` / `time_of_event` (before alert - 1.5 s = normal driving,
alert..event = hazard; `ctrl_*` = the same test on an equally long window of normal driving earlier in the video,
i.e. what random interventions alone would score); Australian clips use the reviewed per-sample labels of the
dataset (`--labels`) and their category (all of a "Normal" clip is normal driving).
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adas_vla.config import Config, load_config  # noqa: E402
from adas_vla.perception.depth import DepthFrame  # noqa: E402
from adas_vla.types import Detection, EgoState, LongAction, SceneContext  # noqa: E402

NEXAR = Path("data/raw/nexar-ai--nexar_collision_prediction/train/positive")
AU = Path("data/raw/qutegocentric--Australian_Roads_Dashcam_Driving")
AU_CLASSES = ("Normal", "Near Crash", "Crash")
PRE_ALERT_S = 10.0  # Nexar: normal driving captured before the alert
AMBIGUOUS_S = 1.5  # ... except the last 1.5 s before it (the hazard may already be visible)
CONTROL_GAP_S = 3.0  # control window (same length as alert..event) ends this long before the ambiguous part


def _clips_nexar(test_percent: int, limit: int | None) -> list[dict]:
    from adas_vla.datasets.nexar import DEFAULT_SPEED_KMH, SCENE_SPEED_KMH, split_for_video

    with (NEXAR / "metadata.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["time_of_alert"] and r["time_of_event"]]
    rows = [r for r in rows if split_for_video(r["file_name"], test_percent) == "test_nexar"
            and (NEXAR / r["file_name"]).exists()]
    clips = []
    for r in rows[:limit]:
        alert, event = float(r["time_of_alert"]), float(r["time_of_event"])
        clips.append({"name": f"nexar_{Path(r['file_name']).stem}", "video": str(NEXAR / r["file_name"]),
                      "kind": "nexar", "ego_kmh": SCENE_SPEED_KMH.get(r["scene"], DEFAULT_SPEED_KMH),
                      "start_s": max(0.0, alert - PRE_ALERT_S), "end_s": event + 0.5,
                      "alert": alert, "event": event})
    return clips


def _clips_au(labels: Path, per_class: int) -> list[dict]:
    from adas_vla.training.data import load_records

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for rec in load_records(labels):
        if rec["id"].startswith("australian_"):
            by_clip[rec["id"].rsplit("_", 1)[0].removeprefix("australian_")].append(rec)
    picked: dict[str, list[dict]] = defaultdict(list)
    for name in sorted(by_clip, key=lambda n: (by_clip[n][0].get("split") != "val", n)):  # val clips first
        video = next((AU / c / f"{name}.mp4" for c in AU_CLASSES if (AU / c / f"{name}.mp4").exists()), None)
        if video is None or len(picked[video.parent.name]) >= per_class:
            continue
        recs = by_clip[name]
        picked[video.parent.name].append({
            "name": f"au_{name}", "video": str(video), "kind": "au_" + video.parent.name.lower().replace(" ", "_"),
            "ego_kmh": recs[0]["ego_speed_kmh"], "start_s": 0.0, "end_s": None,
            "samples": [(int(r["id"].rsplit("_", 1)[1]), r["target"]["longitudinal"]) for r in recs]})
    return [c for cls in AU_CLASSES for c in picked[cls]]


def capture(args) -> None:
    import cv2

    import adas_vla.perception.detector as detector_mod
    from adas_vla.perception.detector import ObjectDetector
    from adas_vla.perception.lanes import build_lane_detector

    detector_mod.is_ego_hood = lambda *a, **k: False  # keep every box: the replay applies the current filter
    cfg = load_config(overrides=args.overrides)
    detector = ObjectDetector(cfg.perception)
    lanes = build_lane_detector(cfg.perception.lane_model, cfg.perception.lane_weights, cfg.perception.device)
    depth = _depth_estimator(cfg) if cfg.perception.depth.enabled else None
    clips = _clips_nexar(args.test_percent, args.nexar) + _clips_au(Path(args.labels), args.au_per_class)
    for spec in args.video:
        path, ego = spec.rsplit(":", 1)
        clips.append({"name": Path(path).stem, "video": path, "kind": "normal", "ego_kmh": float(ego),
                      "start_s": 0.0, "end_s": None})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, clip in enumerate(clips, 1):
        dest = out / f"{clip['name']}.pkl"
        if dest.exists():
            continue
        cap = cv2.VideoCapture(clip["video"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        detector.reset()
        lanes.reset()
        frames, idx, w, h = [], 0, 0, 0
        while True:
            t = idx / fps
            if clip["end_s"] is not None and t > clip["end_s"]:
                break
            if t < clip["start_s"]:
                ok = cap.grab()
            else:
                ok, frame = cap.read()
                if ok:
                    h, w = frame.shape[:2]
                    found = detector(frame)
                    dets = [(d.cls_name, d.conf, d.box, d.track_id, d.attribute) for d in found]
                    rec = {"i": idx, "t": t, "dets": dets, "lanes": lanes(frame)}
                    if depth is not None:
                        rec["depth"] = depth(frame, found).to_dict()
                    frames.append(rec)
            if not ok:
                break
            idx += 1
        cap.release()
        clip.update(fps=fps, width=w, height=h, frames=frames)
        if depth is not None:
            clip["depth_model"] = _depth_name(cfg)
        if "samples" in clip:
            clip["samples"] = [(i / fps, a) for i, a in clip["samples"]]
        with dest.open("wb") as f:
            pickle.dump(clip, f)
        print(f"[{k}/{len(clips)}] {clip['name']}: {len(frames)} frames", flush=True)


def _depth_estimator(cfg: Config):
    from adas_vla.perception.depth import DepthEstimator

    return DepthEstimator(cfg.perception.depth, cfg.camera)


def _depth_name(cfg: Config) -> str:
    d = cfg.perception.depth
    return d.onnx_path if d.backend == "onnx" else d.model


def add_depth(args) -> None:
    """Add the Depth-Anything statistics to existing captures: the videos are read again frame by frame, the stored
    detections are reused (same boxes, same order), the detector and lanes are not re-run."""
    import cv2

    cfg = load_config(overrides=args.overrides)
    cfg.perception.depth.enabled = True
    depth = _depth_estimator(cfg)
    paths = sorted(Path(args.data).glob("*.pkl"))
    for k, path in enumerate(paths, 1):
        with path.open("rb") as f:
            clip = pickle.load(f)
        if not args.force and clip["frames"] and all("depth" in fr for fr in clip["frames"]):
            print(f"[{k}/{len(paths)}] {clip['name']}: depth already present ({clip.get('depth_model')})", flush=True)
            continue
        wanted = {fr["i"]: fr for fr in clip["frames"]}
        cap = cv2.VideoCapture(clip["video"])
        idx, last, done = 0, max(wanted, default=-1), 0
        while idx <= last:
            if idx in wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                fr = wanted[idx]
                dets = [Detection(cls_name=c, conf=conf, box=box, track_id=tid, attribute=attr)
                        for c, conf, box, tid, attr in fr["dets"]]
                fr["depth"] = depth(frame, dets).to_dict()
                done += 1
            elif not cap.grab():
                break
            idx += 1
        cap.release()
        clip["depth_model"] = _depth_name(cfg)
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(clip, f)
        tmp.replace(path)
        print(f"[{k}/{len(paths)}] {clip['name']}: depth on {done}/{len(wanted)} frames", flush=True)


def run_clip(clip: dict, cfg: Config) -> list[tuple[float, LongAction, set[str]]]:
    from adas_vla.control import SafetySupervisor
    from adas_vla.perception import MotionEstimator
    from adas_vla.perception.detector import is_ego_hood

    p = cfg.perception
    motion = MotionEstimator(cfg.camera, cut_in_rate=p.cut_in_rate, cut_in_max_distance_m=p.cut_in_max_distance_m,
                             cut_in_max_pull_away_mps=p.cut_in_max_pull_away_mps, vel_window_s=p.velocity_window_s,
                             depth=p.depth)
    safety = SafetySupervisor(cfg)
    ego = EgoState(clip["ego_kmh"])
    w, h = clip["width"], clip["height"]
    out = []
    for fr in clip["frames"]:
        keep = [k for k, (_, _, box, _, _) in enumerate(fr["dets"]) if not is_ego_hood(box, w, h)]
        dets = [Detection(cls_name=fr["dets"][k][0], conf=fr["dets"][k][1], box=fr["dets"][k][2],
                          track_id=fr["dets"][k][3], attribute=fr["dets"][k][4]) for k in keep]
        depth = None
        if fr.get("depth") is not None and p.depth.mode != "off":
            depth = DepthFrame.from_dict(fr["depth"]).subset(keep)
        motion.update(dets, fr["t"], fr["lanes"], w, h, ego.speed_mps, depth=depth)
        ctx = SceneContext(frame_idx=fr["i"], timestamp_s=fr["t"], width=w, height=h, ego=ego,
                           detections=dets, lanes=fr["lanes"])
        decision, alerts = safety.arbitrate(ctx, None)
        out.append((fr["t"], decision.longitudinal, {a.kind for a in alerts}))
    return out


def _episodes(flags: list[bool]) -> int:
    return sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))


def evaluate(clips: list[dict], cfg: Config) -> dict:
    normal_frames = normal_aeb = normal_fcw = normal_eps = 0
    normal_s = 0.0
    hazard = {"n": 0, "aeb": 0, "warn": 0, "aeb_ctrl": 0, "warn_ctrl": 0}
    samples = {"keep_n": 0, "keep_aeb": 0, "brake_n": 0, "brake_resp": 0}
    for clip in clips:
        res = run_clip(clip, cfg)
        if not res:
            continue
        dt = 1.0 / clip["fps"]
        if clip["kind"] == "nexar":
            normal = [r for r in res if r[0] < clip["alert"] - AMBIGUOUS_S]
            window = [r for r in res if clip["alert"] <= r[0] <= clip["event"]]
            # control: a window of the same length in normal driving (chance level of the hazard numbers)
            c1 = clip["alert"] - AMBIGUOUS_S - CONTROL_GAP_S
            control = [r for r in res if c1 - (clip["event"] - clip["alert"]) <= r[0] <= c1]
            hazard["n"] += 1
            hazard["aeb"] += any("AEB" in r[2] for r in window)
            hazard["warn"] += any(r[2] & {"AEB", "FCW", "PED"} for r in window)
            hazard["aeb_ctrl"] += any("AEB" in r[2] for r in control)
            hazard["warn_ctrl"] += any(r[2] & {"AEB", "FCW", "PED"} for r in control)
        elif clip["kind"] in ("au_normal", "normal"):
            normal = res
        else:
            normal = []
        normal_frames += len(normal)
        normal_s += len(normal) * dt
        aeb = ["AEB" in r[2] for r in normal]
        normal_aeb += sum(aeb)
        normal_fcw += sum("FCW" in r[2] or "AEB" in r[2] for r in normal)
        normal_eps += _episodes(aeb)
        for t, label in clip.get("samples", []):
            r = min(res, key=lambda r: abs(r[0] - t))
            if label in ("ACCELERATE", "KEEP"):
                samples["keep_n"] += 1
                samples["keep_aeb"] += "AEB" in r[2]
            elif label in ("BRAKE", "STOP", "EMERGENCY_BRAKE"):
                samples["brake_n"] += 1
                samples["brake_resp"] += r[1].rank >= LongAction.DECELERATE.rank
    pct = lambda a, b: 100.0 * a / b if b else float("nan")  # noqa: E731
    return {
        "normal_min": normal_s / 60,
        "aeb_time_%": pct(normal_aeb, normal_frames),
        "aeb_per_min": normal_eps / (normal_s / 60) if normal_s else float("nan"),
        "fcw+aeb_time_%": pct(normal_fcw, normal_frames),
        "nexar_aeb_%": pct(hazard["aeb"], hazard["n"]),
        "ctrl_aeb_%": pct(hazard["aeb_ctrl"], hazard["n"]),
        "nexar_warn_%": pct(hazard["warn"], hazard["n"]),
        "ctrl_warn_%": pct(hazard["warn_ctrl"], hazard["n"]),
        "au_keep_aeb_%": pct(samples["keep_aeb"], samples["keep_n"]),
        "au_brake_resp_%": pct(samples["brake_resp"], samples["brake_n"]),
        "n": f"{hazard['n']} nexar / {samples['keep_n']}+{samples['brake_n']} au samples",
    }


def replay(args) -> None:
    clips = []
    for path in sorted(Path(args.data).glob("*.pkl")):
        with path.open("rb") as f:
            clips.append(pickle.load(f))
    variants = args.variant or ["current:"]
    with_depth = sum(1 for c in clips for fr in c["frames"] if fr.get("depth") is not None)
    total = sum(len(c["frames"]) for c in clips)
    models = sorted({c["depth_model"] for c in clips if c.get("depth_model")})
    print(f"frames: {total}, with depth statistics: {with_depth} ({', '.join(models) or 'none'}); "
          "variants with perception.depth.mode != off use them, pinhole elsewhere")
    rows = []
    for spec in variants:
        name, _, sets = spec.partition(":")
        cfg = load_config(overrides=[s for s in sets.split(",") if s])
        rows.append((name, evaluate(clips, cfg)))
    keys = [k for k in rows[0][1] if k != "n"]
    print(f"{'variant':<16}" + "".join(f"{k:>16}" for k in keys))
    for name, m in rows:
        print(f"{name:<16}" + "".join(f"{m[k]:>16.2f}" for k in keys))
    print("normal driving: Nexar before alert - 1.5 s + AU 'Normal' clips; hazard: Nexar alert..event;",
          rows[0][1]["n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="run detector + lanes once, store raw per-frame outputs")
    c.add_argument("--out", default="outputs/gate_replay")
    c.add_argument("--nexar", type=int, default=None, help="max Nexar test videos (default: all)")
    c.add_argument("--test-percent", type=int, default=13, help="same held-out split as build-dataset nexar")
    c.add_argument("--labels", default="data/ds_v2/labels.jsonl", help="reviewed labels of the AU clips")
    c.add_argument("--au-per-class", type=int, default=8)
    c.add_argument("--video", action="append", default=[], help="extra normal-driving video: PATH:EGO_KMH")
    c.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. perception.depth.enabled=true")
    d = sub.add_parser("depth", help="add Depth-Anything statistics to existing captures (videos re-read)")
    d.add_argument("data")
    d.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. perception.depth.model=... or perception.depth.backend=onnx")
    d.add_argument("--force", action="store_true", help="recompute clips that already have depth statistics")
    r = sub.add_parser("replay", help="re-run geometry + motion + safety gate on captured outputs")
    r.add_argument("data")
    r.add_argument("--variant", action="append", help="NAME:key=value,key=value (config overrides)")
    args = ap.parse_args()
    {"capture": capture, "depth": add_depth, "replay": replay}[args.cmd](args)


if __name__ == "__main__":
    main()
