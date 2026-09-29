"""Deterministic safety gate between the VLM recommendation and the vehicle.

The VLM only *recommends* a meta-action. This module decides: it falls back to a rule-based ACC policy
when the recommendation is missing or stale, enforces AEB/FCW/VRU braking, caps speed behind a lead
vehicle, and blocks lane changes that need driver confirmation. It must stay simple enough to verify.
"""

from __future__ import annotations

from dataclasses import replace

from ..config import Config
from ..types import VRU_CLASSES, Alert, DrivingDecision, LatAction, LongAction, RiskLevel, SceneContext


class SafetySupervisor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def acc_speed_cap_kmh(self, ctx: SceneContext) -> float | None:
        """Max speed that keeps the desired time gap to the lead object (constant time-gap ACC law)."""
        lead = ctx.lead_object()
        if lead is None or lead.distance_m is None:
            return None
        c = self.cfg.control
        return max(0.0, (lead.distance_m - c.standstill_gap_m) / c.time_gap_s) * 3.6

    def rule_decision(self, ctx: SceneContext) -> DrivingDecision:
        """Baseline policy used whenever no fresh VLM recommendation exists."""
        ego = ctx.ego.speed_kmh
        target = self.cfg.control.cruise_speed_kmh
        cap = self.acc_speed_cap_kmh(ctx)
        reason = "cruise"
        if cap is not None and cap < target:
            target, reason = cap, "following lead object (ACC)"
        if target < ego * 0.5:
            action = LongAction.BRAKE
        elif target < ego - 3:
            action = LongAction.DECELERATE
        elif target > ego + 3:
            action = LongAction.ACCELERATE
        else:
            action = LongAction.KEEP
        return DrivingDecision(
            longitudinal=action, lateral=LatAction.KEEP_LANE, target_speed_kmh=target,
            risk_level=RiskLevel.LOW, reason=reason, source="rules",
            frame_idx=ctx.frame_idx, timestamp_s=ctx.timestamp_s,
        )

    def is_fresh(self, decision: DrivingDecision | None, ctx: SceneContext) -> bool:
        return decision is not None and (
            ctx.timestamp_s - decision.timestamp_s <= self.cfg.vlm.max_decision_age_s)

    @staticmethod
    def _at_least(decision: DrivingDecision, action: LongAction, target_kmh: float, reason: str) -> DrivingDecision:
        """Escalate braking to at least `action`; never relaxes a stronger decision."""
        if decision.longitudinal.rank >= action.rank:
            return replace(decision, target_speed_kmh=min(decision.target_speed_kmh, target_kmh))
        return replace(decision, longitudinal=action, target_speed_kmh=min(decision.target_speed_kmh, target_kmh),
                       risk_level=RiskLevel.HIGH, reason=reason, source="safety")

    def arbitrate(self, ctx: SceneContext, vlm_decision: DrivingDecision | None
                  ) -> tuple[DrivingDecision, list[Alert]]:
        s = self.cfg.safety
        alerts: list[Alert] = []
        ego = ctx.ego.speed_kmh
        ego_mps = ctx.ego.speed_mps

        if self.is_fresh(vlm_decision, ctx):
            decision = replace(vlm_decision, frame_idx=ctx.frame_idx)
        else:
            decision = self.rule_decision(ctx)

        # 1. Longitudinal collision checks on the nearest in-path object.
        lead = ctx.lead_object()
        if lead is not None and lead.distance_m is not None:
            ttc = lead.ttc_s
            gap_s = lead.distance_m / ego_mps if ego_mps > 0.5 else None
            if (ttc is not None and ttc < s.aeb_ttc_s) or (lead.distance_m < s.min_distance_m and ego_mps > 1):
                alerts.append(Alert("AEB", f"Emergency brake: {lead.cls_name} at {lead.distance_m:.0f} m"
                                    + (f", TTC {ttc:.1f} s" if ttc is not None else ""), "critical"))
                decision = replace(decision, longitudinal=LongAction.EMERGENCY_BRAKE, lateral=LatAction.KEEP_LANE,
                                   target_speed_kmh=0.0, risk_level=RiskLevel.HIGH, source="safety",
                                   reason=f"AEB: {lead.cls_name} too close")
            elif (ttc is not None and ttc < s.fcw_ttc_s) or (gap_s is not None and gap_s < s.min_time_gap_s):
                alerts.append(Alert("FCW", f"Forward collision warning: {lead.cls_name} at {lead.distance_m:.0f} m",
                                    "warning"))
                decision = self._at_least(decision, LongAction.DECELERATE, ego * 0.8, "FCW: closing on lead object")

        # 2. Vulnerable road users in the ego path.
        for det in ctx.detections:
            if det.cls_name in VRU_CLASSES and det.in_ego_path and det.distance_m is not None \
                    and det.distance_m < s.vru_brake_distance_m:
                alerts.append(Alert("PED", f"{det.cls_name} in path at {det.distance_m:.0f} m", "critical"))
                decision = self._at_least(decision, LongAction.BRAKE, 0.0, f"{det.cls_name} in ego path")
                break

        # 3. Red traffic light ahead: inform only (monocular distance to lights is unreliable).
        if any(d.cls_name == "traffic light" and d.attribute == "red" for d in ctx.detections):
            alerts.append(Alert("RED_LIGHT", "Red traffic light ahead", "info"))

        # 4. Speed envelope: legal/configured max and ACC gap cap.
        cap = min(s.max_speed_kmh, self.cfg.control.cruise_speed_kmh * 1.1)
        acc_cap = self.acc_speed_cap_kmh(ctx)
        if acc_cap is not None:
            cap = min(cap, acc_cap)
        if decision.target_speed_kmh > cap:
            decision = replace(decision, target_speed_kmh=cap)
            if decision.longitudinal is LongAction.ACCELERATE:
                decision = replace(decision, longitudinal=LongAction.KEEP if cap >= ego - 3 else LongAction.DECELERATE)

        # 5. Lane changes need driver confirmation on an L2 system (no side/rear sensing here).
        if decision.lateral in (LatAction.CHANGE_LEFT, LatAction.CHANGE_RIGHT) and not s.allow_lane_change:
            side = "left" if decision.lateral is LatAction.CHANGE_LEFT else "right"
            alerts.append(Alert("LANE_CHANGE_SUGGESTED", f"Lane change {side} suggested - confirm to proceed", "info"))
            decision = replace(decision, lateral=LatAction.KEEP_LANE)

        # 6. Lane departure warning.
        offset = ctx.lanes.offset_norm
        if offset is not None and abs(offset) > s.ldw_offset:
            alerts.append(Alert("LDW", f"Lane departure {'right' if offset > 0 else 'left'}", "warning"))

        return decision, alerts
