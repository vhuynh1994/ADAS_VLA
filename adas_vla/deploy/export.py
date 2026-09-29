"""Export the perception detector as a static-shape ONNX + calibration set for QAIRT quantization.

Output folder:
  <name>.onnx          static [1, 3, H, W] input, no NMS in graph (NMS/decoding stay on the CPU)
  calib/*.raw          float32 NCHW RGB in [0, 1], letterboxed like Ultralytics preprocessing
  input_list.txt       one raw file per line, the format qairt-quantizer expects for --input_list
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from ..config import Config

DEFAULT_CALIB_SOURCES = ["data/samples/highway_traffic.mp4", "data/samples/highway_short.mp4"]


def letterbox(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    scale = min(h / frame.shape[0], w / frame.shape[1])
    nh, nw = round(frame.shape[0] * scale), round(frame.shape[1] * scale)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def write_calibration(sources: list[str], out_dir: Path, h: int, w: int, n: int = 100) -> int:
    calib_dir = out_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for src in sources:
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        for idx in np.linspace(0, max(total - 1, 0), num=max(1, n // len(sources)), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
    lines = []
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(letterbox(frame, h, w), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        path = calib_dir / f"calib_{i:04d}.raw"
        np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).tofile(path)
        lines.append(str(path.resolve()))
    (out_dir / "input_list.txt").write_text("\n".join(lines) + "\n")
    return len(lines)


def export_detector(cfg: Config, out: str, imgsz: tuple[int, int] = (384, 640), qnn_arch: str | None = None,
                    calib_sources: list[str] | None = None) -> None:
    from ultralytics import YOLO

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = imgsz
    model = YOLO(cfg.perception.detector_model)
    onnx_path = Path(model.export(format="onnx", imgsz=[h, w], opset=17, simplify=True, dynamic=False,
                                  half=False, nms=False))
    target = out_dir / onnx_path.name
    shutil.move(str(onnx_path), target)
    print(f"ONNX: {target}  (input [1,3,{h},{w}], NMS on host)")

    sources = [s for s in (calib_sources or DEFAULT_CALIB_SOURCES) if Path(s).exists()]
    if sources:
        count = write_calibration(sources, out_dir, h, w)
        print(f"Calibration: {count} samples -> {out_dir / 'input_list.txt'}")
    else:
        print("No calibration videos found (run scripts/download_samples.sh or pass your own ADAS clips).")

    if qnn_arch:
        # Ultralytics' QNN export wraps a context binary for onnxruntime-qnn (Linux/Windows hosts).
        qnn = model.export(format="qnn", name=str(qnn_arch), imgsz=[h, w])
        print(f"Ultralytics QNN export (HTP v{qnn_arch}): {qnn}")

    print("\nNext: follow docs/DEPLOY_SA8797P.md (qairt-converter -> qairt-quantizer -> context binary).")
