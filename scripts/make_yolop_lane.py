"""Lane-only YOLOP at a non-square input: keep the lane-line head, drop the detection and drivable-area heads.

The released YOLOP ONNX files (hustvl/YOLOP, MIT) are square (320/640/1280), so a 16:9 dashcam frame letterboxed into
640x640 is 44% gray padding, and the drivable-area head costs as much as the lane head (26.5% of the conv MACs each)
although the pipeline never reads it. The graph is size-agnostic (Focus slices with step 2, Resize by scale 2), so
the input can be set to any H x W that are multiples of 32. The YOLOP paper (arXiv 2108.11250, sec. 4.1) resizes
BDD100K frames from 1280x720 to 640x384.

  python scripts/make_yolop_lane.py --src models/yolop/yolop-640-640.onnx \
      --out models/yolop/yolop-lane-384x640.onnx --height 384 --width 640 \
      --data data/ds_v3 --raw-dir outputs/deploy_sa8650/yolop_lane --calib 200 --eval 20

With --data, also writes QAIRT inputs preprocessed exactly like perception/lanes.py (letterbox with pad 114, RGB,
ImageNet mean/std, float32 NCHW): calib/ from the train split, eval/ from held-out frames (val, Nexar test), each
spread over sources and videos, with input lists and eval/frames.json naming the source images.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import shape_inference, utils

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # as YolopLaneDetector
STD = np.array([0.229, 0.224, 0.225], np.float32)


def make_lane_model(src: str, out: str, height: int, width: int) -> None:
    if height % 32 or width % 32:
        raise SystemExit("height and width must be multiples of 32")
    tmp = Path(out).with_suffix(".tmp.onnx")
    utils.extract_model(src, str(tmp), ["images"], ["lane_line_seg"])
    model = onnx.load(str(tmp))
    tmp.unlink()
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_value, dims[3].dim_value = height, width
    del model.graph.value_info[:]
    for o in model.graph.output:  # re-derived by shape inference
        o.type.tensor_type.ClearField("shape")
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"{out}: images [1,3,{height},{width}] -> lane_line_seg {shape}, {len(model.graph.node)} nodes")


def pick_frames(data: Path, splits: tuple[str, ...], n: int, seed: int) -> list[dict]:
    """n records of the given splits, round-robin over sources and over videos within a source."""
    overlay = data / "labels.splits.json"
    overlay = json.loads(overlay.read_text()) if overlay.exists() else {}
    by_src: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for line in (data / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        if overlay.get(r["id"], r.get("split")) in splits:
            by_src[r["source"]][r.get("group", r["id"])].append(r)
    rng = random.Random(seed)
    queues = {}
    for src, groups in sorted(by_src.items()):
        order = list(groups.values())
        rng.shuffle(order)
        for g in order:
            rng.shuffle(g)
        queues[src] = [g[i] for i in range(max(map(len, order))) for g in order if i < len(g)]
    picked, i = [], 0
    while len(picked) < n and any(i < len(q) for q in queues.values()):
        picked += [q[i] for q in queues.values() if i < len(q)][:n - len(picked)]
        i += 1
    return picked


def write_raws(data: Path, records: list[dict], out_dir: Path, height: int, width: int) -> None:
    from adas_vla.deploy.export import letterbox

    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, r in enumerate(records):
        frame = cv2.imread(str(data / r["image"]))
        rgb = cv2.cvtColor(letterbox(frame, height, width), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        path = out_dir / f"lane_{i:04d}.raw"
        np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None]).tofile(path)
        lines.append(str(path.resolve()))
    (out_dir / "input_list.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "frames.json").write_text(json.dumps([{"raw": Path(l).name, "image": r["image"], "source": r["source"]}
                                                     for l, r in zip(lines, records)], indent=1))
    print(f"{out_dir}: {len(lines)} inputs " + str({s: sum(r['source'] == s for r in records)
                                                     for s in sorted({r['source'] for r in records})}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="models/yolop/yolop-640-640.onnx")
    p.add_argument("--out", default="models/yolop/yolop-lane-384x640.onnx")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--data", help="dataset dir with labels.jsonl (+ labels.splits.json) for calibration / eval inputs")
    p.add_argument("--raw-dir", default="outputs/deploy_sa8650/yolop_lane")
    p.add_argument("--calib", type=int, default=200)
    p.add_argument("--eval", type=int, default=20)
    a = p.parse_args()
    make_lane_model(a.src, a.out, a.height, a.width)
    if a.data:
        data, raw = Path(a.data), Path(a.raw_dir)
        write_raws(data, pick_frames(data, ("train",), a.calib, 0), raw / "calib", a.height, a.width)
        write_raws(data, pick_frames(data, ("val", "test_nexar"), a.eval, 1), raw / "eval", a.height, a.width)


if __name__ == "__main__":
    main()
