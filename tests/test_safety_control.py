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
    s = SafetySupervisor(Config())
    for t in (10.0, 10.1, 10.2):  # AEB acts once its trigger held for aeb_confirm_s
        d, alerts = s.arbitrate(ctx_with([car(12, ttc=1.0)], t=t), vlm(LongAction.ACCELERATE, t=t, lat_a=LatAction.NUDGE_LEFT))
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


def no_confirm() -> Config:
    cfg = Config()
    cfg.safety.aeb_confirm_s = 0.0
    return cfg


def test_aeb_holds_after_trigger_and_uses_hysteresis():
    s = SafetySupervisor(no_confirm())
    d, _ = s.arbitrate(ctx_with([car(12, ttc=1.0)], t=10.0), vlm(LongAction.KEEP, t=10.0))
    assert d.longitudinal is LongAction.EMERGENCY_BRAKE
    # TTC 1.8 s: above the 1.5 s trigger but below the release threshold (1.5 x 1.3) -> still AEB
    d, _ = s.arbitrate(ctx_with([car(20, ttc=1.8)], t=10.1), vlm(LongAction.KEEP, t=10.1))
    assert d.longitudinal is LongAction.EMERGENCY_BRAKE
    # detection dropped for a frame: held for 0.5 s after the last trigger
    d, alerts = s.arbitrate(ctx_with([], t=10.3), vlm(LongAction.KEEP, t=10.3))
    assert d.longitudinal is LongAction.EMERGENCY_BRAKE and alerts[0].kind == "AEB" and "holding" in alerts[0].message
    # AEB released; FCW (triggered by the same frames) is held for 1 s
    d, alerts = s.arbitrate(ctx_with([], t=10.7), vlm(LongAction.KEEP, t=10.7))
    assert d.longitudinal is LongAction.DECELERATE and [a.kind for a in alerts] == ["FCW"]
    d, alerts = s.arbitrate(ctx_with([], t=11.2), vlm(LongAction.KEEP, t=11.2))
    assert d.longitudinal is LongAction.KEEP and not alerts


def test_no_hold_behaves_frame_by_frame():
    cfg = no_confirm()
    cfg.safety.aeb_hold_s = cfg.safety.fcw_hold_s = 0.0
    cfg.safety.hysteresis = 1.0
    s = SafetySupervisor(cfg)
    d, _ = s.arbitrate(ctx_with([car(12, ttc=1.0)], t=10.0), vlm(LongAction.KEEP, t=10.0))
    assert d.longitudinal is LongAction.EMERGENCY_BRAKE
    d, alerts = s.arbitrate(ctx_with([], t=10.1), vlm(LongAction.KEEP, t=10.1))
    assert d.longitudinal is LongAction.KEEP and not alerts


def test_reset_forgets_held_interventions():
    s = SafetySupervisor(no_confirm())
    s.arbitrate(ctx_with([car(12, ttc=1.0)], t=10.0), vlm(LongAction.KEEP, t=10.0))
    s.reset()  # new video: the timeline restarts
    d, alerts = s.arbitrate(ctx_with([], t=0.0), vlm(LongAction.KEEP, t=0.0))
    assert d.longitudinal is LongAction.KEEP and not alerts


def test_cutting_in_vehicle_is_the_lead():
    cut = Detection(cls_name="car", conf=0.9, box=(900, 400, 1000, 480), distance_m=15, in_ego_path=False,
                    cutting_in=True)
    d, _ = SafetySupervisor(Config()).arbitrate(ctx_with([cut], speed=60), vlm(LongAction.ACCELERATE, 90))
    assert d.longitudinal is LongAction.DECELERATE  # the ACC gap cap applies to the cutting-in car
    assert abs(d.target_speed_kmh - (15 - 5.0) / 1.8 * 3.6) < 1e-6


def moving(dist, closing):
    d = car(dist, ttc=dist / closing)
    d.closing_speed_mps = closing
    return d


def test_aeb_confirmation_ignores_one_frame_phantoms():
    cfg = Config()
    cfg.safety.aeb_confirm_s, cfg.safety.aeb_confirm_gap_s = 0.15, 0.0
    s = SafetySupervisor(cfg)
    d, alerts = s.arbitrate(ctx_with([moving(12, 12)], t=10.0), vlm(LongAction.KEEP, t=10.0))
    assert d.longitudinal is LongAction.DECELERATE and [a.kind for a in alerts] == ["FCW"]  # not confirmed yet
    d, _ = s.arbitrate(ctx_with([], t=10.05), vlm(LongAction.KEEP, t=10.05))  # gone: it was a one-off
    assert d.longitudinal is not LongAction.EMERGENCY_BRAKE
    for t in (10.1, 10.2):  # a real target keeps triggering
        d, _ = s.arbitrate(ctx_with([moving(12, 12)], t=t), vlm(LongAction.KEEP, t=t))
    assert d.longitudinal is LongAction.DECELERATE  # 0.1 s of triggering so far
    d, alerts = s.arbitrate(ctx_with([moving(11, 12)], t=10.25), vlm(LongAction.KEEP, t=10.25))
    assert d.longitudinal is LongAction.EMERGENCY_BRAKE and alerts[0].kind == "AEB"


def test_aeb_confirmation_counts_through_short_dropouts():
    """Crash-clip regression: a car cutting in at 5 m is missed by the detector for a few frames at a time; the
    confirmation must not restart on every dropout, but a long gap does restart it."""
    cfg = Config()
    cfg.safety.aeb_confirm_s, cfg.safety.aeb_confirm_gap_s = 0.1, 0.05
    s = SafetySupervisor(cfg)
    seen = []
    for t, dets in ((10.0, [moving(5, 5)]), (10.02, []), (10.04, [moving(5, 5)]), (10.06, []), (10.08, [moving(5, 5)]),
                    (10.1, [moving(5, 5)])):
        d, _ = s.arbitrate(ctx_with(dets, t=t), vlm(LongAction.KEEP, t=t))
        seen.append(d.longitudinal is LongAction.EMERGENCY_BRAKE)
    assert seen == [False] * 5 + [True]  # 0.1 s since the first trigger, dropouts of 0.02 s tolerated
    s = SafetySupervisor(cfg)
    for t, dets in ((10.0, [moving(5, 5)]), (10.02, []), (10.09, [moving(5, 5)]), (10.12, [moving(5, 5)])):
        d, _ = s.arbitrate(ctx_with(dets, t=t), vlm(LongAction.KEEP, t=t))
    assert d.longitudinal is not LongAction.EMERGENCY_BRAKE  # the 0.07 s gap restarted the count at 10.09
