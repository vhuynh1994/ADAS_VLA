"""VLM on the SA8650P board (`vlm.backend: board`): InternVL3.5-2B served by board/vlm_genie_server.cpp.

Same interface as VisionLanguageModel.decide(): the project's decision prompt is rendered with InternVL's chat
template on the PC (tokenizer files only, no weights), split around the image, and sent with the 448x448 image
(NCHW float32, ImageNet normalisation: the AI Hub bundle's input) to the board, which returns the generated text.
The board always generates the whole answer (Genie 1.18 ignores an early stop), so the reason is kept: it
costs no latency here and the HUD shows it.
"""

from __future__ import annotations

import re
import socket
import struct
import threading
import time

import numpy as np

from ..config import VLMConfig, resolve_model
from ..types import DrivingDecision
from .parser import parse_decision
from .prompts import DECISION_PREFIX, THINKING_PREFIX, decision_messages, split_thinking, thinking_messages

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MENU_COPY = re.compile(r'"([A-Z_]+) = [^"]*"')  # zero-shot the base model may copy "KEEP = hold current speed"


def pixel_values(image) -> bytes:
    """PIL image or BGR array -> the bundle's 1x3x448x448 float32 input."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1])
    rgb = np.asarray(image.convert("RGB").resize((448, 448), Image.BICUBIC), np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32).tobytes()


def split_prompt(text: str) -> tuple[str, str]:
    """Chat-template text with one image placeholder -> (text up to <img>, text from </img>)."""
    i = text.index("<IMG_CONTEXT>")
    return text[:i] + "<img>", "</img>" + text[i + len("<IMG_CONTEXT>"):]


class BoardVLM:
    def __init__(self, cfg: VLMConfig):
        from transformers import AutoProcessor

        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(resolve_model(cfg.board_processor))
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self.last_timing: dict[str, float] = {}

    def _connect(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.create_connection((self.cfg.board_host, self.cfg.board_port), timeout=60)
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
        req = (b"VLM1" + blob(before.encode()) + blob(after.encode()) + blob(pixel_values(image))
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

        msgs = decision_messages(Image.new("RGB", (448, 448)), context_text, ego_speed_kmh, cruise_speed_kmh,
                                 self.cfg.language)
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
