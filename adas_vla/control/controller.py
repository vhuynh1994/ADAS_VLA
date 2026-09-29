"""Low-level controller: turns an arbitrated meta-action into throttle / brake / steer."""

from __future__ import annotations

from ..config import ControlConfig
from ..types import ControlCommand, DrivingDecision, LatAction, LongAction, SceneContext


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Controller:
    """P-controllers for speed tracking and lane keeping, plus per-action lateral bias."""

    def __init__(self, cfg: ControlConfig):
        self.cfg = cfg

    def __call__(self, decision: DrivingDecision, ctx: SceneContext) -> ControlCommand:
        long_a, lat_a = decision.longitudinal, decision.lateral
        target = decision.target_speed_kmh
        ego = ctx.ego.speed_kmh

        if long_a is LongAction.EMERGENCY_BRAKE:
            throttle, brake, target = 0.0, 1.0, 0.0
        elif long_a is LongAction.STOP:
            throttle, brake, target = 0.0, _clip(0.3 + 0.01 * ego, 0.3, 0.8), 0.0
        else:
            err = target - ego
            throttle = _clip(self.cfg.kp_speed * err, 0.0, 1.0) if err > 0 else 0.0
            brake = _clip(-self.cfg.kp_speed * 0.5 * err, 0.0, 1.0) if err < 0 else 0.0
            if long_a is LongAction.BRAKE:
                brake = max(brake, 0.4)
                throttle = 0.0

        steer = 0.0
        offset = ctx.lanes.offset_norm
        if offset is not None:
            steer = -self.cfg.kp_steer * offset  # lane-keeping assist: steer back to lane center
        steer += {
            LatAction.NUDGE_LEFT: -self.cfg.nudge_bias, LatAction.NUDGE_RIGHT: self.cfg.nudge_bias,
            LatAction.CHANGE_LEFT: -self.cfg.lane_change_bias, LatAction.CHANGE_RIGHT: self.cfg.lane_change_bias,
        }.get(lat_a, 0.0)

        return ControlCommand(throttle=throttle, brake=brake, steer=_clip(steer, -1.0, 1.0),
                              target_speed_kmh=target, longitudinal=long_a, lateral=lat_a)
