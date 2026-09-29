from adas_vla.config import Config
from adas_vla.control import Controller, SafetySupervisor
from adas_vla.types import (Detection, DrivingDecision, EgoState, LaneInfo, LatAction, LongAction, RiskLevel,
                            SceneContext)


def ctx_with(dets=(), speed=50.0, t=10.0, lanes=None):
    return SceneContext(frame_idx=0, timestamp_s=t, width=1280, height=720, ego=EgoState(speed),
                        detections=list(dets), lanes=lanes or LaneInfo(image_width=1280))


def car(dist, ttc=None, in_path=True, cls="car"):
    return Detection(cls_name=cls, conf=0.9, box=(600, 400, 700, 480), distance_m=dist, ttc_s=ttc,
                     in_ego_path=in_path)


def vlm(long_a, speed=60.0, t=10.0, lat_a=LatAction.KEEP_LANE):
    return DrivingDecision(longitudinal=long_a, lateral=lat_a, target_speed_kmh=speed, risk_level=RiskLevel.LOW,
                           timestamp_s=t)


def test_rules_used_when_vlm_missing_or_stale():
    s = SafetySupervisor(Config())
    d, _ = s.arbitrate(ctx_with(), None)
    assert d.source == "rules" and d.longitudinal is LongAction.ACCELERATE  # 50 -> cruise 60
    d, _ = s.arbitrate(ctx_with(t=10.0), vlm(LongAction.KEEP, t=5.0))
    assert d.source == "rules"


def test_fresh_vlm_decision_passes_through():
    d, alerts = SafetySupervisor(Config()).arbitrate(ctx_with(), vlm(LongAction.KEEP, 55))
    assert d.longitudinal is LongAction.KEEP and d.source == "vlm" and not alerts


def test_aeb_overrides_vlm():
    d, alerts = SafetySupervisor(Config()).arbitrate(ctx_with([car(12, ttc=1.0)]), vlm(LongAction.ACCELERATE, lat_a=LatAction.NUDGE_LEFT))
    assert d.longitudinal is LongAction.EMERGENCY_BRAKE and d.lateral is LatAction.KEEP_LANE and d.target_speed_kmh == 0 and d.source == "safety"
    assert alerts[0].kind == "AEB" and alerts[0].level == "critical"


def test_fcw_escalates_but_never_relaxes():
    s = SafetySupervisor(Config())
    d, alerts = s.arbitrate(ctx_with([car(40, ttc=2.0)]), vlm(LongAction.KEEP))
    assert d.longitudinal is LongAction.DECELERATE and any(a.kind == "FCW" for a in alerts)
    d, _ = s.arbitrate(ctx_with([car(40, ttc=2.0)]), vlm(LongAction.STOP, 0))
    assert d.longitudinal is LongAction.STOP


def test_small_time_gap_triggers_fcw():
    # 10 m at 50 km/h = 0.72 s headway < 0.8 s
    _, alerts = SafetySupervisor(Config()).arbitrate(ctx_with([car(10)]), vlm(LongAction.KEEP))
    assert any(a.kind == "FCW" for a in alerts)


def test_pedestrian_in_path_forces_brake():
    d, alerts = SafetySupervisor(Config()).arbitrate(ctx_with([car(10, cls="person")], speed=30),
                                                     vlm(LongAction.KEEP, 30))
    assert d.longitudinal is LongAction.BRAKE and any(a.kind == "PED" for a in alerts)


def test_acc_caps_speed_behind_lead():
    s = SafetySupervisor(Config())
    d, _ = s.arbitrate(ctx_with([car(30)], speed=50), vlm(LongAction.ACCELERATE, 90))
    cap = (30 - 5.0) / 1.8 * 3.6  # 50 km/h
    assert abs(d.target_speed_kmh - cap) < 1e-6
    assert d.longitudinal is not LongAction.ACCELERATE


def test_lane_change_needs_confirmation():
    d, alerts = SafetySupervisor(Config()).arbitrate(ctx_with(), vlm(LongAction.KEEP, lat_a=LatAction.CHANGE_LEFT))
    assert d.lateral is LatAction.KEEP_LANE
    assert any(a.kind == "LANE_CHANGE_SUGGESTED" for a in alerts)
    cfg = Config()
    cfg.safety.allow_lane_change = True
    d, _ = SafetySupervisor(cfg).arbitrate(ctx_with(), vlm(LongAction.KEEP, lat_a=LatAction.CHANGE_LEFT))
    assert d.lateral is LatAction.CHANGE_LEFT


def test_ldw_alert():
    lanes = LaneInfo(left_fit=(0.0, 500.0), right_fit=(0.0, 900.0), y_top=430, y_bottom=720, image_width=1280)
    # lane center 700, ego center 640 -> offset -0.3 (fine); shift lanes so offset is large
    _, alerts = SafetySupervisor(Config()).arbitrate(ctx_with(lanes=lanes), vlm(LongAction.KEEP))
    assert not any(a.kind == "LDW" for a in alerts)
    lanes = LaneInfo(left_fit=(0.0, 620.0), right_fit=(0.0, 1020.0), y_top=430, y_bottom=720, image_width=1280)
    _, alerts = SafetySupervisor(Config()).arbitrate(ctx_with(lanes=lanes), vlm(LongAction.KEEP))
    assert any(a.kind == "LDW" for a in alerts)


def test_controller_emergency_and_tracking():
    c = Controller(Config().control)
    cmd = c(vlm(LongAction.EMERGENCY_BRAKE, 0), ctx_with())
    assert cmd.brake == 1.0 and cmd.throttle == 0.0
    cmd = c(vlm(LongAction.ACCELERATE, 70), ctx_with(speed=50))
    assert cmd.throttle > 0 and cmd.brake == 0
    cmd = c(vlm(LongAction.DECELERATE, 30), ctx_with(speed=50))
    assert cmd.brake > 0 and cmd.throttle == 0


def test_controller_lane_keeping_steers_back():
    lanes = LaneInfo(left_fit=(0.0, 440.0), right_fit=(0.0, 740.0), y_top=430, y_bottom=720, image_width=1280)
    # lane center 590 < ego 640 -> ego is right of center -> steer left (negative)
    cmd = Controller(Config().control)(vlm(LongAction.KEEP), ctx_with(lanes=lanes))
    assert cmd.steer < 0


def test_long_action_rank_orders_braking():
    assert LongAction.ACCELERATE.rank < LongAction.KEEP.rank < LongAction.DECELERATE.rank < LongAction.BRAKE.rank
    assert LongAction.BRAKE.rank < LongAction.STOP.rank < LongAction.EMERGENCY_BRAKE.rank


def test_controller_lateral_bias():
    c = Controller(Config().control)
    left = c(vlm(LongAction.KEEP, lat_a=LatAction.NUDGE_LEFT), ctx_with())
    right = c(vlm(LongAction.KEEP, lat_a=LatAction.CHANGE_RIGHT), ctx_with())
    assert left.steer < 0 < right.steer
