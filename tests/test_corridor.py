import pytest

from adas_vla.config import Config, load_config
from adas_vla.perception import MotionEstimator, focal_length_px
from adas_vla.perception.geometry import lane_horizon, lateral_offset_m, metric_corridor
from adas_vla.types import Detection, LaneInfo

W, H = 1280, 720
# a wide road: lane lines through (100, 600) and (1180, 600) meeting at (640, 360), i.e. horizon at 0.5 H
WIDE = LaneInfo(left_fit=(-2.25, 1450.0), right_fit=(2.25, -170.0), y_top=400.0, y_bottom=720.0, image_width=W)


def test_lane_horizon_is_the_vanishing_point():
    assert lane_horizon(WIDE, H) == pytest.approx(360.0)
    assert lane_horizon(LaneInfo(image_width=W), H) is None
    assert lane_horizon(LaneInfo((1.0, 0.0), (1.0, 100.0), image_width=W), H) is None  # parallel
    assert lane_horizon(LaneInfo((-1.0, 700.0), (1.0, 580.0), image_width=W), H) is None  # meet at y = 60


def test_metric_corridor_caps_the_lane_corridor():
    half = 1.2 * (560 - 360) / 1.3  # 1.2 m at the row where the road is (560 - 360) px below the horizon
    assert metric_corridor(560, (100, 1200), W, H, 360, 1.3, 1.2) == pytest.approx((640 - half, 640 + half))
    assert metric_corridor(560, (700, 1200), W, H, 360, 1.3, 1.2) == pytest.approx((700, 640 + half))
    # lane corridor off to one side (a lane line of the next lane): the axis strip
    assert metric_corridor(560, (1000, 1200), W, H, 360, 1.3, 1.2) == pytest.approx((640 - half, 640 + half))
    assert lateral_offset_m(640 + half, 560, W, H, 360, 1.3) == pytest.approx(1.2)


def test_corridor_cap_keeps_the_next_lane_out_but_not_pedestrians():
    def in_path(cls, cap):
        est = MotionEstimator(Config().camera, corridor_half_width_m=cap)
        det = Detection(cls, 0.9, (900, 520, 1040, 600))  # 1.4..2.2 m right of our axis, inside the wide lanes
        est.update([det], 0.0, WIDE, W, H)
        return det.in_ego_path

    assert in_path("car", None) and not in_path("car", 1.2)
    assert in_path("person", 1.2)  # pedestrians can step into our path: the lane corridor still applies


def test_truck_crossing_ahead_fills_the_narrow_corridor():
    est = MotionEstimator(Config().camera, corridor_half_width_m=1.2)
    det = Detection("truck", 0.9, (200, 360, 1200, 420))  # far, 1000 px wide, the corridor only ~110 px there
    est.update([det], 0.0, WIDE, W, H)
    assert det.in_ego_path


def test_horizon_relaxes_to_the_configured_row_without_lanes():
    est = MotionEstimator(Config().camera)
    lanes = LaneInfo(left_fit=(-1.111, 1000.0), right_fit=(1.111, 280.0), image_width=W)  # meet at y ~ 324
    assert est.horizon_row(lanes, 0.0, H) == pytest.approx(324, abs=1)
    assert est.horizon_row(LaneInfo(image_width=W), 0.5, H) == pytest.approx(324 + (360 - 324) * 0.393, abs=1)
    assert est.horizon_row(LaneInfo(image_width=W), 10.0, H) == pytest.approx(360, abs=0.1)


def _approach(est, x1, x2, dists, y2=450, lanes=WIDE, dx=0.0):
    f = focal_length_px(W, Config().camera.hfov_deg)
    det = None
    for i, dist in enumerate(dists):
        h_px = f * 1.5 / dist
        det = Detection("car", 0.9, (x1 - i * dx, y2 - h_px, x2 - i * dx, y2), track_id=3)
        est.update([det], i * 0.1, lanes, W, H, 60 / 3.6)
    return det


def test_oncoming_car_off_axis_is_not_the_lead():
    closing_30mps = [80.0, 77.0, 74.0]  # while we drive 16.7 m/s
    det = _approach(MotionEstimator(Config().camera, dist_tau_s=0.0), 700, 760, closing_30mps)
    assert det.oncoming and det.in_ego_path  # inside the wide lane corridor, 0.9 m right of our axis
    det = _approach(MotionEstimator(Config().camera, dist_tau_s=0.0, exclude_oncoming=True), 700, 760, closing_30mps)
    assert det.oncoming and not det.in_ego_path
    det = _approach(MotionEstimator(Config().camera, dist_tau_s=0.0, exclude_oncoming=True), 610, 670, closing_30mps)
    assert det.oncoming and det.in_ego_path  # head-on: still the lead


def test_oncoming_car_turning_across_our_path_stays_a_threat():
    est = MotionEstimator(Config().camera, dist_tau_s=0.0, exclude_oncoming=True)
    det = _approach(est, 1000, 1100, [30.0, 27.0, 24.0, 21.0], y2=600, lanes=LaneInfo(image_width=W), dx=40)
    assert det.oncoming and det.cutting_in


def test_oncoming_car_whose_side_face_grows_is_not_cutting_in():
    """Approaching on its own side (left-hand traffic: on our right), an oncoming car's near box edge grows toward
    our corridor as its side face comes into view while its far edge holds: no cut-in, so not the lead (the false
    AEB of the UK dashcam video). With cut_in_oncoming_outer off it was flagged and stayed in the path."""
    def run(outer):
        est = MotionEstimator(Config().camera, dist_tau_s=0.0, exclude_oncoming=True, cut_in_oncoming_outer=outer)
        f = focal_length_px(W, Config().camera.hfov_deg)
        det = None
        for i, dist in enumerate([30.0, 27.0, 24.0, 21.0]):
            det = Detection("car", 0.9, (1000 - 40 * i, 600 - f * 1.5 / dist, 1100, 600), track_id=3)
            est.update([det], i * 0.1, LaneInfo(image_width=W), W, H, 60 / 3.6)
        return det

    assert run(None).cutting_in
    det = run(0.5)
    assert det.oncoming and not det.cutting_in and not det.in_ego_path


def test_oncoming_car_turning_across_stays_a_threat_with_the_far_edge_check():
    est = MotionEstimator(Config().camera, dist_tau_s=0.0, exclude_oncoming=True, cut_in_oncoming_outer=0.5)
    det = _approach(est, 1000, 1100, [30.0, 27.0, 24.0, 21.0], y2=600, lanes=LaneInfo(image_width=W), dx=40)
    assert det.oncoming and det.cutting_in  # the whole box moves toward our path


def test_defaults_enable_the_metric_corridor_and_oncoming_exclusion():
    est = MotionEstimator.from_config(load_config())
    assert est.corridor_half_width_m == 1.2 and est.exclude_oncoming and est.cut_in_oncoming_outer == 0.5
