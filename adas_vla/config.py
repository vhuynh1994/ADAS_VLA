"""Typed configuration loaded from YAML, with `key.sub=value` overrides."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_model(model_id: str) -> str:
    """Prefer an existing local directory, then models/<org>--<name> (scripts/fetch_hf.py), then the HF hub."""
    for candidate in (Path(model_id), PROJECT_ROOT / model_id, MODELS_DIR / model_id.replace("/", "--")):
        if (candidate / "config.json").exists():
            return str(candidate)
    return model_id


@dataclass
class CameraConfig:
    hfov_deg: float = 60.0  # horizontal field of view of the front camera
    mount_height_m: float = 1.3


@dataclass
class PerceptionConfig:
    detector_model: str = "yolo11s.pt"
    device: str = "cuda:0"
    conf: float = 0.35
    imgsz: int = 640
    track: bool = True
    classes: list[str] = field(default_factory=lambda: [
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign",
    ])
    lane_detection: bool = True
    lane_model: str = "yolop"  # yolop (CNN, robust to faded markings) | classic (Canny + Hough)
    lane_weights: str = "models/yolop/yolop-640-640.onnx"
    # Cut-in detection: an adjacent vehicle within cut_in_max_distance_m whose lateral gap to the ego corridor
    # shrinks faster than cut_in_rate (corridor widths per second) is reported as cutting in.
    cut_in_rate: float = 0.25
    cut_in_max_distance_m: float = 30.0
    cut_in_max_pull_away_mps: float = 1.0  # a vehicle moving away from us faster than this is not a cut-in threat
    velocity_window_s: float = 0.5  # closing speed = least-squares distance slope over this window


@dataclass
class VLMConfig:
    enabled: bool = True
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter_path: str | None = None  # LoRA adapter produced by `adas-vla train`
    quantization: str = "4bit"  # none | 4bit | 8bit
    quantize_vision: bool = False  # keep the vision encoder in bf16 (faster + more accurate on the PC)
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"
    trigger: str = "event"  # interval: every N frames | event: also on scene changes (new lead, VRU, ...)
    min_interval_frames: int = 5  # event trigger never fires faster than this
    # Fixed [width, height] fed to the VLM (static shape, as required on the Hexagon NPU).
    # Use multiples of 28 for Qwen2.5-VL, 32 for Qwen3-VL. null -> keep aspect, cap at image_max_side.
    image_size: list[int] | None = field(default_factory=lambda: [560, 308])
    image_max_side: int = 640
    max_new_tokens: int = 64
    generate_reason: bool = False  # False: stop after the decision fields (latency); True: also the reason text
    # > 0: feed the frame this many seconds earlier together with the current one as a 2-frame video. Qwen2.5-VL
    # packs 2 frames into one temporal patch, so this costs no extra visual tokens; 0.5 s matches its default
    # 2 fps video sampling. 0 = single image. Needs a model fine-tuned with the same setting.
    prev_frame_s: float = 0.0
    # greedy: most likely action | cautious: escalate to DECELERATE / BRAKE when the probability of needing
    # at least that much braking reaches the threshold (trades some false slowdowns for fewer missed ones)
    # | cautious_gated: cautious only when perception corroborates a hazard (types.context_has_hazard_cue)
    action_policy: str = "greedy"
    cautious_tau_decel: float = 0.35
    cautious_tau_brake: float = 0.30
    temperature: float = 0.0  # 0 = greedy decoding
    language: str = "en"  # en | vi: language of free-text fields (scene, reason, chat)
    mode: str = "sync"  # sync: deterministic, for offline video | async: background thread, for live camera
    every_n_frames: int = 15
    max_decision_age_s: float = 2.0  # older VLM decisions are ignored in favour of the rule-based policy


@dataclass
class LLMConfig:
    """Text-only LLM that explains ADAS interventions from structured logs (mature path on the NPU)."""

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    quantization: str = "4bit"  # none | 4bit | 8bit
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"
    max_new_tokens: int = 160
    temperature: float = 0.0
    language: str = "vi"


@dataclass
class SafetyConfig:
    aeb_ttc_s: float = 1.5
    fcw_ttc_s: float = 2.7
    min_distance_m: float = 4.0
    min_time_gap_s: float = 0.8  # headway below this triggers FCW
    vru_brake_distance_m: float = 15.0  # pedestrian/cyclist in path closer than this -> brake
    ldw_offset: float = 0.6  # |lane offset| above this triggers lane departure warning
    allow_lane_change: bool = False  # L2 ADAS: lane changes need driver confirmation
    max_speed_kmh: float = 130.0
    # Interventions are Schmitt triggers: once active, AEB / FCW release only when their thresholds are cleared
    # by `hysteresis` (x1.3), and stay active at least `*_hold_s` after the last trigger (no single-frame flapping).
    aeb_hold_s: float = 0.5
    fcw_hold_s: float = 1.0
    hysteresis: float = 1.3
    aeb_confirm_s: float = 0.2  # AEB acts once triggered this long (target confirmation, see scripts/gate_replay.py)


@dataclass
class ControlConfig:
    cruise_speed_kmh: float = 60.0
    time_gap_s: float = 1.8  # ACC desired headway
    standstill_gap_m: float = 5.0
    kp_speed: float = 0.08
    kp_steer: float = 0.5
    nudge_bias: float = 0.1
    lane_change_bias: float = 0.25


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    ego_speed_kmh: float = 60.0  # used when no CAN bus speed is available (video files)


def _build(cls, data: dict):
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if dataclasses.is_dataclass(hints[f.name]) and isinstance(value, dict):
            value = _build(hints[f.name], value)
        kwargs[f.name] = value
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def apply_override(data: dict, override: str) -> None:
    """Apply a `section.key=value` override; value is parsed as YAML (numbers, bools, null)."""
    if "=" not in override:
        raise ValueError(f"Override must look like key.sub=value, got: {override}")
    key, raw = override.split("=", 1)
    node = data
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    for ov in overrides or []:
        apply_override(data, ov)
    return _build(Config, data)
