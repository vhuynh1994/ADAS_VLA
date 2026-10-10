"""Core data types shared across perception, reasoning and control."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum


class LongAction(str, Enum):
    """Longitudinal meta-action (speed). Ordered by braking strength."""

    ACCELERATE = "ACCELERATE"
    KEEP = "KEEP"
    DECELERATE = "DECELERATE"
    BRAKE = "BRAKE"
    STOP = "STOP"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"  # issued only by the safety gate

    @property
    def rank(self) -> int:
        """Braking strength: higher = brakes harder. Used to detect under-braking."""
        return list(LongAction).index(self)


class LatAction(str, Enum):
    """Lateral meta-action (lane)."""

    KEEP_LANE = "KEEP_LANE"
    NUDGE_LEFT = "NUDGE_LEFT"
    NUDGE_RIGHT = "NUDGE_RIGHT"
    CHANGE_LEFT = "CHANGE_LEFT"
    CHANGE_RIGHT = "CHANGE_RIGHT"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
VRU_CLASSES = {"person", "bicycle"}  # vulnerable road users


@dataclass
class Detection:
    cls_name: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    track_id: int | None = None
    distance_m: float | None = None
    closing_speed_mps: float | None = None  # > 0 means the object is getting closer
    ttc_s: float | None = None  # time-to-collision
    in_ego_path: bool = False
    position: str = "ahead"  # ahead | front-left | front-right
    attribute: str | None = None  # e.g. traffic light color

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2

    oncoming: bool = False  # moving toward the ego vehicle (e.g. opposite carriageway)
    cutting_in: bool = False  # adjacent vehicle moving into the ego corridor (see MotionEstimator)

    @property
    def threatening(self) -> bool:
        """Objects we could hit: in the ego path, or about to enter it."""
        return self.in_ego_path or self.cutting_in

    def describe(self) -> str:
        name = self.cls_name if not self.attribute else f"{self.cls_name} ({self.attribute})"
        parts = [name, self.position]
        if self.distance_m is not None:
            parts.append(f"~{self.distance_m:.0f} m")
        if self.in_ego_path:
            parts.append("IN EGO LANE")
        elif self.cutting_in:
            parts.append(f"CUTTING IN from the {'left' if self.position == 'front-left' else 'right'}")
        if self.oncoming:
            parts.append("oncoming traffic")
        elif self.threatening and self.closing_speed_mps is not None and abs(self.closing_speed_mps) > 0.5:
            # Relative motion only matters for objects we could hit; for others it misleads the VLM.
            verb = "closing in" if self.closing_speed_mps > 0 else "pulling away"
            parts.append(f"{verb} at {abs(self.closing_speed_mps) * 3.6:.0f} km/h")
        if self.ttc_s is not None and self.threatening:
            parts.append(f"TTC {self.ttc_s:.1f} s")
        return ", ".join(parts)


@dataclass
class LaneInfo:
    """Lane boundaries as lines x = m * y + b in image coordinates."""

    left_fit: tuple[float, float] | None = None
    right_fit: tuple[float, float] | None = None
    y_top: float = 0.0
    y_bottom: float = 0.0
    image_width: int = 0

    @staticmethod
    def x_at(fit: tuple[float, float], y: float) -> float:
        m, b = fit
        return m * y + b

    @property
    def valid(self) -> bool:
        return self.left_fit is not None and self.right_fit is not None

    def bounds_at(self, y: float) -> tuple[float, float] | None:
        if not self.valid:
            return None
        return self.x_at(self.left_fit, y), self.x_at(self.right_fit, y)

    @property
    def offset_norm(self) -> float | None:
        """Ego offset from lane center, normalised by half the lane width.

        0 = centered, +1 = on the right lane line, -1 = on the left lane line.
        Assumes the camera is mounted on the vehicle's center line.
        """
        bounds = self.bounds_at(self.y_bottom)
        if bounds is None:
            return None
        left, right = bounds
        half_width = (right - left) / 2
        if half_width <= 1:
            return None
        return (self.image_width / 2 - (left + right) / 2) / half_width

    def describe(self) -> str:
        if not self.valid:
            seen = [s for s, f in (("left", self.left_fit), ("right", self.right_fit)) if f is not None]
            return f"only {seen[0]} lane line visible" if seen else "lane lines not detected"
        off = self.offset_norm or 0.0
        if abs(off) < 0.15:
            where = "centered in lane"
        else:
            where = f"drifting {'right' if off > 0 else 'left'} ({abs(off) * 100:.0f}% of half lane)"
        return f"both lane lines visible, {where}"


@dataclass
class EgoState:
    speed_kmh: float = 0.0

    @property
    def speed_mps(self) -> float:
        return self.speed_kmh / 3.6


@dataclass
class SceneContext:
    frame_idx: int
    timestamp_s: float
    width: int
    height: int
    ego: EgoState
    detections: list[Detection] = field(default_factory=list)
    lanes: LaneInfo = field(default_factory=LaneInfo)

    def lead_object(self) -> Detection | None:
        """Nearest object in the ego path, or cutting into it (vehicle or vulnerable road user)."""
        candidates = [
            d for d in self.detections
            if d.threatening and d.distance_m is not None
            and (d.cls_name in VEHICLE_CLASSES or d.cls_name in VRU_CLASSES)
        ]
        return min(candidates, key=lambda d: d.distance_m) if candidates else None

    def summary_text(self, max_objects: int = 8) -> str:
        """Compact textual description of perception output, used in the VLM prompt."""
        lines = [f"Lane: {self.lanes.describe()}."]
        ranked = sorted(self.detections, key=lambda d: (not d.threatening, d.distance_m or 1e9))
        if ranked:
            lines.append("Detected objects (monocular distance estimates):")
            lines += [f"- {d.describe()}" for d in ranked[:max_objects]]
            if len(ranked) > max_objects:
                lines.append(f"- ... and {len(ranked) - max_objects} more")
        else:
            lines.append("Detected objects: none.")
        return "\n".join(lines)


_DISTANCE_RE = re.compile(r"~(\d+(?:\.\d+)?) m")


def context_has_hazard_cue(context_text: str, max_distance_m: float = 40.0) -> bool:
    """Does the perception summary (`summary_text()`) show a reason to slow down?

    True for an object in the ego lane or cutting into it within `max_distance_m` that is not pulling away,
    or a red traffic light. Parsed from the text so the same rule applies online (`VisionLanguageModel.decide`)
    and offline on logged eval records (`scripts/sweep_policy.py`).
    """
    for line in context_text.splitlines():
        if "traffic light (red)" in line:
            return True
        if ("IN EGO LANE" not in line and "CUTTING IN" not in line) or "pulling away" in line:
            continue
        m = _DISTANCE_RE.search(line)
        if m is None or float(m.group(1)) <= max_distance_m:
            return True
    return False


@dataclass
class DrivingDecision:
    longitudinal: LongAction
    lateral: LatAction
    target_speed_kmh: float
    risk_level: RiskLevel = RiskLevel.LOW
    reason: str = ""
    source: str = "vlm"  # vlm | rules | safety
    frame_idx: int = -1
    timestamp_s: float = 0.0
    latency_s: float = 0.0
    action_probs: dict[str, float] | None = None  # VLM probability of each longitudinal action

    @property
    def label(self) -> str:
        return f"{self.longitudinal.value} / {self.lateral.value}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["longitudinal"] = self.longitudinal.value
        d["lateral"] = self.lateral.value
        d["risk_level"] = self.risk_level.value
        return d

    def target_dict(self) -> dict:
        """The fields the VLM is asked to produce, in output order (used as training target)."""
        return {
            "longitudinal": self.longitudinal.value,
            "lateral": self.lateral.value,
            "target_speed_kmh": round(self.target_speed_kmh),
            "risk": self.risk_level.value,
            "reason": self.reason,
        }


@dataclass
class ControlCommand:
    throttle: float  # 0..1
    brake: float  # 0..1
    steer: float  # -1 (full left) .. +1 (full right)
    target_speed_kmh: float
    longitudinal: LongAction
    lateral: LatAction


@dataclass
class Alert:
    kind: str  # AEB | FCW | PED | LDW | RED_LIGHT | LANE_CHANGE_SUGGESTED
    message: str
    level: str = "warning"  # info | warning | critical


@dataclass
class FrameResult:
    context: SceneContext
    decision: DrivingDecision  # final, after safety arbitration
    vlm_decision: DrivingDecision | None  # latest raw VLM output (may be stale)
    command: ControlCommand
    alerts: list[Alert] = field(default_factory=list)
    perception_ms: float = 0.0
    timing: dict[str, float] = field(default_factory=dict)  # per-stage ms, see adas_vla/timing.py
    vlm_age_s: float | None = None  # age of the VLM decision in effect, on the frame timeline (what the gate uses)
    vlm_age_wall_s: float | None = None  # same, wall clock since that frame's capture (real age on a live camera)
