"""VLM on the SA8650P board (`vlm.backend: board`): a Genie VLM served by board/vlm_genie_server.cpp, either
InternVL3.5-2B (AI Hub bundle) or the fine-tuned Qwen2.5-VL adas-vlm-v3 built by scripts/v3_qnn.py; which one
follows from `vlm.board_processor` (config.json model_type).

Same interface as VisionLanguageModel.decide(): the project's decision prompt is rendered with the model's chat
template on the PC (tokenizer files only, no weights), split around the image, and sent with the image to the
board, which returns the generated text. InternVL: 448x448 uint8 RGB (protocol VLM2; the board applies the
ImageNet normalisation of the bundle's float32 input). Qwen2.5-VL: uint8 RGB at vlm.image_size (560x308; VLM3, the
board cuts the 14x14 patches like the Qwen2-VL processor; server started with --qwen). Where the answer stops is
set by the server's generator config (board/vlm_server.sh): `fast` stops before "reason" (action only), `greedy`
writes the reason too.
"""

from __future__ import annotations

import json
import re
import socket
import struct
import threading
import time

from pathlib import Path

import numpy as np

from ..config import VLMConfig, resolve_model
from ..types import DrivingDecision
from .parser import parse_decision
from .prompts import DECISION_PREFIX, THINKING_PREFIX, decision_messages, split_thinking, thinking_messages

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MENU_COPY = re.compile(r'"([A-Z_]+) = [^"]*"')  # zero-shot the base model may copy "KEEP = hold current speed"


def rgb_resized(image, size: tuple[int, int]) -> np.ndarray:
    """PIL image or BGR array -> uint8 RGB [h, w, 3] at size (w, h), PIL bicubic (what the PC sends)."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1])
    image = image.convert("RGB")
    if image.size != tuple(size):
        image = image.resize(tuple(size), Image.BICUBIC)
    return np.asarray(image, np.uint8)


def rgb_448(image) -> np.ndarray:
    """PIL image or BGR array -> 448x448x3 uint8 RGB (InternVL, protocol VLM2)."""
    return rgb_resized(image, (448, 448))


def pixel_values(image) -> bytes:
    """PIL image or BGR array -> the bundle's 1x3x448x448 float32 input (protocol VLM1; VLM2 does this on the board)."""
    rgb = rgb_448(image).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32).tobytes()


def split_prompt(text: str) -> tuple[str, str]:
    """Chat-template text with one image placeholder -> (text up to <img>, text from </img>) for InternVL, or
    (text up to <|vision_start|>, text from <|vision_end|>) for Qwen2.5-VL."""
    qwen = "<|vision_start|><|image_pad|><|vision_end|>"
    if qwen in text:
        i = text.index(qwen)
        return text[:i] + "<|vision_start|>", text[i + len(qwen) - len("<|vision_end|>"):]
    i = text.index("<IMG_CONTEXT>")
    return text[:i] + "<img>", "</img>" + text[i + len("<IMG_CONTEXT>"):]


class BoardVLM:
    def __init__(self, cfg: VLMConfig):
        from transformers import AutoProcessor

        self.cfg = cfg
        path = resolve_model(cfg.board_processor)
        self.processor = AutoProcessor.from_pretrained(path)
        model_type = json.loads((Path(path) / "config.json").read_text()).get("model_type", "")
        self.qwen = model_type.startswith("qwen2")
        self.size = tuple(cfg.image_size or (560, 308)) if self.qwen else (448, 448)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self.last_timing: dict[str, float] = {}

    def _connect(self, wait_s: float = 30.0) -> socket.socket:
        """Connect to the server; while it is still loading (connection refused) retry for up to wait_s."""
        deadline = time.monotonic() + wait_s
        while self._sock is None:
            try:
                self._sock = socket.create_connection((self.cfg.board_host, self.cfg.board_port), timeout=60)
            except ConnectionRefusedError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)
                continue
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self._sock

    def _recv(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("board VLM server closed the connection")
            buf += chunk
        return buf

    def generate(self, image, before: str, after: str, max_new_tokens: int) -> str:
        blob = lambda b: struct.pack("<I", len(b)) + b  # noqa: E731
        magic = b"VLM3" if self.qwen else b"VLM2"
        req = (magic + blob(before.encode()) + blob(after.encode()) + blob(rgb_resized(image, self.size).tobytes())
               + struct.pack("<I", max_new_tokens))
        with self._lock:
            try:
                sock = self._connect()
                sock.sendall(req)
                head = self._recv(4 + 4 + 12 + 4 + 4)
            except OSError:
                self._sock = None
                raise
            magic, status, image_ms, first_ms, total_ms, tokens, n = struct.unpack("<4si3fII", head)
            text = self._recv(n).decode("utf-8", "replace")
        if magic != b"VLR1" or status != 0:
            raise RuntimeError(f"board VLM error (status {status})")
        self.last_timing = {"image_ms": image_ms, "first_token_ms": first_ms, "total_ms": total_ms, "tokens": tokens}
        return text

    def decide(self, image, context_text: str, ego_speed_kmh: float, cruise_speed_kmh: float,
               max_speed_kmh: float = 130.0, prev_image=None) -> tuple[DrivingDecision | None, str, float]:
        from PIL import Image

        msgs = decision_messages(Image.new("RGB", self.size), context_text, ego_speed_kmh, cruise_speed_kmh,
                                 self.cfg.language, self.cfg.explain)
        thinking = ""
        if self.cfg.thinking:
            msgs = thinking_messages(msgs)
        before, after = split_prompt(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        t0 = time.perf_counter()
        if self.cfg.thinking:  # the server config caps the length (max-num-tokens)
            thinking, raw = split_thinking(THINKING_PREFIX + self.generate(image, before, after + THINKING_PREFIX, 0))
        else:
            raw = DECISION_PREFIX + self.generate(image, before, after + DECISION_PREFIX, self.cfg.max_new_tokens)
        latency = time.perf_counter() - t0
        raw = MENU_COPY.sub(r'"\1"', raw)
        if "}" not in raw and "," in raw:  # cut inside a field (max_new_tokens): keep the complete ones
            raw = raw[:raw.rfind(",")] + "}"
        decision = parse_decision(raw, ego_speed_kmh, max_speed_kmh)
        if decision is not None:
            decision.latency_s = latency
            decision.thinking = thinking
        return decision, raw, latency

    def chat(self, *args, **kwargs) -> str:
        raise NotImplementedError("the board VLM serves driving decisions only")
