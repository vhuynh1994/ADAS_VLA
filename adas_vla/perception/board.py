"""Perception on the SA8650P HTP through board/vla_stream_server (`perception.backend: board`).

The board runs the detector (YOLO11s W8A8) and the lane model (lane-only YOLOP 384x640 W8A16) on one JPEG of the
letterboxed frame and returns boxes (Ultralytics-equivalent filtering + NMS) and the lane mask. ByteTrack, lane line
fitting, distances / TTC, the safety gate, the controller and the HUD stay on the PC, unchanged.

Protocol: see the header of board/vla_stream_server.cpp.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import PerceptionConfig
from ..types import Detection
from .detector import is_ego_hood, traffic_light_color
from .lanes import LaneDetector

MEAN = (0.485, 0.456, 0.406)  # YOLOP input normalization (RGB)
STD = (0.229, 0.224, 0.225)


@dataclass
class TensorInfo:
    name: str
    dtype: int
    scale: float
    offset: int
    dims: list[int]

    @property
    def bits(self) -> int:
        return {0x08: 8, 0x16: 16}.get(self.dtype & 0xFF, 0)


def parse_header(lines: list[str]) -> dict[str, dict]:
    """Server header lines -> {role: {"graph": name, "in": [TensorInfo], "out": [TensorInfo]}}."""
    if not lines or lines[0] != "VLASTREAM 1":
        raise ConnectionError(f"not a vla_stream_server: {lines[:1]}")
    models, cur = {}, None
    for line in lines[1:]:
        parts = line.split()
        if parts[0] == "MODEL":
            cur = models[parts[1]] = {"graph": parts[2], "in": [], "out": []}
        elif parts[0] in ("IN", "OUT"):
            rank = int(parts[5])
            cur[parts[0].lower()].append(TensorInfo(parts[1], int(parts[2], 16), float(parts[3]), int(parts[4]),
                                                    [int(d) for d in parts[6:6 + rank]]))
    return models


def quant_lut(t: TensorInfo, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)) -> np.ndarray:
    """uint16 [3, 256]: pixel value -> quantized input for real = (pixel / 255 - mean[c]) / std[c].

    The real value is computed in float32 in that order, like the PC preprocessing (lanes.py, make_yolop_lane.py), and
    quantized with the QNN SDK's datautil::floatToTfN (encodingMin = offset * scale in float, C round), which is what
    qnn-net-run does with float inputs: the board then sees exactly the input qnn-net-run would (QNN: real = (q + offset)
    * scale). Plain round(real / scale) - offset differs by 1 on ~0.3% of 16-bit inputs.
    """
    pix = np.arange(256, dtype=np.float32)
    real = (pix[None, :] / np.float32(255.0) - np.asarray(mean, np.float32)[:, None]) / np.asarray(std, np.float32)[:, None]
    top = 2.0 ** t.bits - 1
    enc_min = float(np.float32(t.offset) * np.float32(t.scale))
    enc_range = (top + t.offset) * float(np.float32(t.scale)) - enc_min
    v = top * (real.astype(np.float64) - enc_min) / enc_range
    q = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5))
    return np.clip(q, 0, top).astype(np.uint16)


def decode_runs(runs: np.ndarray, h: int, w: int) -> np.ndarray:
    """Run lengths alternating background / lane (starting with background) -> uint8 mask h x w (0/1)."""
    values = np.arange(len(runs)) % 2
    return np.repeat(values.astype(np.uint8), runs.astype(np.int64)).reshape(h, w)


class BoardLink:
    """One TCP connection to vla_stream_server; `infer(frame)` returns boxes and the lane mask in frame pixels."""

    def __init__(self, host: str, port: int, class_names: dict[int, str], classes: list[str], conf: float,
                 iou: float = 0.7, max_det: int = 300, jpeg_quality: int = 90, timeout_s: float = 10.0):
        self.jpeg_quality = jpeg_quality
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.models = parse_header(self._read_header())
        det_in, lane_in = self.models["det"]["in"][0], self.models["lane"]["in"][0]
        self.h, self.w = det_in.dims[2], det_in.dims[3]  # NCHW
        num_classes = next(t for t in self.models["det"]["out"] if t.name == "scores").dims[1]
        wanted = set(classes)
        mask = np.array([class_names.get(i) in wanted for i in range(num_classes)], np.uint8)
        setup = (quant_lut(det_in).tobytes() + quant_lut(lane_in, MEAN, STD).tobytes()
                 + struct.pack("<ffI", conf, iou, max_det) + mask.tobytes())
        self._request(3, setup)
        self.last_timing: dict[str, float] = {}

    def _read_header(self) -> list[str]:
        buf = b""
        while not buf.endswith(b"END\n"):
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("board closed the connection during the header")
            buf += chunk
        return buf.decode().splitlines()[:-1]

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("board closed the connection")
            buf += chunk
        return bytes(buf)

    def _request(self, kind: int, payload: bytes) -> tuple[tuple, bytes]:
        self.sock.sendall(struct.pack("<II", kind, len(payload)) + payload)
        head = struct.unpack("<8I", self._recv_exact(32))
        if head[0] != 0:
            raise RuntimeError(f"board request kind {kind} failed with status {head[0]}")
        return head, b""  # payload read by the caller (its size depends on the request kind)

    def letterbox(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        from ..deploy.export import letterbox

        h, w = frame_bgr.shape[:2]
        scale = min(self.h / h, self.w / w)
        nh, nw = round(h * scale), round(w * scale)
        return letterbox(frame_bgr, self.h, self.w), scale, (self.h - nh) // 2, (self.w - nw) // 2

    def infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> boxes [N, 6] (x1 y1 x2 y2 score class) in frame pixels, lane mask (frame size, uint8 0/1)."""
        t0 = time.perf_counter()
        canvas, scale, top, left = self.letterbox(frame_bgr)
        ok, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        t1 = time.perf_counter()
        head, _ = self._request(2, jpeg.tobytes())
        _, recv_us, prep_us, det_us, lane_us, post_us, nboxes, nruns = head
        body = self._recv_exact(nboxes * 24 + nruns * 4)
        t2 = time.perf_counter()
        boxes = np.frombuffer(body, np.float32, nboxes * 6).reshape(nboxes, 6).copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
        h, w = frame_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        mask = decode_runs(np.frombuffer(body, np.uint32, nruns, nboxes * 24), self.h, self.w)
        nh, nw = round(h * scale), round(w * scale)
        mask = cv2.resize(mask[top:top + nh, left:left + nw], (w, h), interpolation=cv2.INTER_NEAREST)
        board_compute_ms = (prep_us + det_us + lane_us + post_us) / 1000
        self.last_timing = {"det_pre": (t1 - t0) * 1000, "board_prep": prep_us / 1000, "board_det": det_us / 1000,
                            "board_lane": lane_us / 1000, "board_post": post_us / 1000,
                            "net": (t2 - t1) * 1000 - board_compute_ms}  # transfer both ways + board receive
        return boxes, mask

    def close(self) -> None:
        try:
            self.sock.sendall(struct.pack("<II", 0, 0))
        except OSError:
            pass
        self.sock.close()


class _Boxes:
    """The part of ultralytics' Boxes that BYTETracker reads (numpy, boolean indexing)."""

    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return _Boxes(self.data[idx])

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5]

    @property
    def xywh(self) -> np.ndarray:
        xy = (self.data[:, :2] + self.data[:, 2:4]) / 2
        return np.concatenate([xy, self.data[:, 2:4] - self.data[:, :2]], 1)


def _coco_names() -> dict[int, str]:
    import yaml
    from ultralytics.utils import ROOT

    return yaml.safe_load(Path(ROOT / "cfg/datasets/coco.yaml").read_text())["names"]


class BoardDetector:
    """Same interface as ObjectDetector (call -> detections, reset, last_timing), with the models on the board."""

    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg
        self.names = _coco_names()
        self.link = BoardLink(cfg.board_host, cfg.board_port, self.names, cfg.classes, cfg.conf,
                              jpeg_quality=cfg.board_jpeg_quality)
        self.lane_mask: np.ndarray | None = None  # mask of the frame of the last call, for BoardLaneDetector
        self.last_timing: dict[str, float] = {}
        self._tracker = None
        self.reset()

    def reset(self) -> None:
        if self.cfg.track:
            from types import SimpleNamespace

            import yaml
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import ROOT

            args = yaml.safe_load(Path(ROOT / "cfg/trackers/bytetrack.yaml").read_text())
            self._tracker = BYTETracker(SimpleNamespace(**args))  # as model.track(persist=True)

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        boxes, self.lane_mask = self.link.infer(frame_bgr)
        self.last_timing = dict(self.link.last_timing)
        if self._tracker is not None:
            # same as Ultralytics' track: keep tracked boxes only, [x1 y1 x2 y2 id score cls idx]
            tracks = self._tracker.update(_Boxes(boxes))
            rows = [(t[:4], float(t[5]), int(t[6]), int(t[4])) for t in tracks]
        else:
            rows = [(b[:4], float(b[4]), int(b[5]), None) for b in boxes]
        h, w = frame_bgr.shape[:2]
        detections = []
        for box, conf, cls, tid in rows:
            if is_ego_hood(box, w, h):
                continue
            name = self.names[cls]
            det = Detection(cls_name=name, conf=conf, box=tuple(float(v) for v in box), track_id=tid)
            if name == "traffic light":
                det.attribute = traffic_light_color(frame_bgr, det.box)
            detections.append(det)
        return detections

    def close(self) -> None:
        self.link.close()


class BoardLaneDetector(LaneDetector):
    """Lane lines fitted (as on the PC) on the lane mask the board computed for the same frame."""

    def __init__(self, detector: BoardDetector, **kwargs):
        super().__init__(**kwargs)
        self.detector = detector

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        h = frame_bgr.shape[0]
        mask = self.detector.lane_mask
        if mask is None:
            return np.zeros(frame_bgr.shape[:2], np.uint8)
        mask = mask * np.uint8(255)
        mask[:int(self.roi_top * h)] = 0  # same ROI as YolopLaneDetector
        return mask
