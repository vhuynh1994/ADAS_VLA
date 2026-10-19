"""Bird's-eye view: projection consistent with perception/geometry.py, objects and lanes land where expected."""

import numpy as np
import pytest

from adas_vla.bev import THREAT, BEVRenderer, object_xz, pitch_error_range, road_point
from adas_vla.config import CameraConfig
from adas_vla.hud import HUD
from adas_vla.perception.geometry import focal_length_px, ground_distance, lateral_offset_m
from adas_vla.types import (ControlCommand, Detection, DrivingDecision, EgoState, FrameResult, LaneInfo, LatAction,
                            LongAction, SceneContext)

W, H, HORIZON, CAM_H = 1280, 720, 360.0, 1.3
F = focal_length_px(W, 60.0)


def lane_fit(x_m: float) -> tuple[float, float]:
    """Image line x = m*y + b of the straight road line at lateral offset x_m (flat road, horizon at HORIZON)."""
    return x_m / CAM_H, W / 2 - x_m * HORIZON / CAM_H


def test_road_point_matches_geometry():
    for x, y in [(640, 500), (300, 600), (1100, 420)]:
        X, Z = road_point(x, y, W, H, HORIZON, CAM_H, F)
        assert Z == pytest.approx(ground_distance(y, F, H, CAM_H, horizon_frac=HORIZON / H))
        assert X == pytest.approx(lateral_offset_m(x, y, W, H, HORIZON, CAM_H))
    assert road_point(640, HORIZON + 5, W, H, HORIZON, CAM_H, F) is None  # too close to the horizon


def test_straight_lane_line_projects_to_constant_offset():
    m, b = lane_fit(-1.75)
    for y in np.linspace(400, 720, 9):
        X, _ = road_point(m * y + b, y, W, H, HORIZON, CAM_H, F)
        assert X == pytest.approx(-1.75)


def test_object_position_from_distance():
    centred = Detection("car", 0.9, (600, 300, 680, 360), distance_m=20.0)
    assert object_xz(centred, W, F) == pytest.approx((0.0, 20.0))
    right = Detection("car", 0.9, (900, 300, 980, 360), distance_m=20.0)
    x, z = object_xz(right, W, F)
    assert z == 20.0 and x == pytest.approx((940 - 640) * 20.0 / F)
    assert object_xz(Detection("car", 0.9, (0, 0, 10, 10)), W, F) is None  # no distance estimate


def _scene(**kw) -> SceneContext:
    lanes = LaneInfo(left_fit=lane_fit(-1.75), right_fit=lane_fit(1.75), y_top=420, y_bottom=H, image_width=W)
    car = Detection("car", 0.9, (600, 330, 680, 380), distance_m=20.0, in_ego_path=True, ttc_s=5.0)
    return SceneContext(frame_idx=0, timestamp_s=0.0, width=W, height=H, ego=EgoState(60), detections=[car],
                        lanes=lanes, horizon_y=kw.get("horizon_y", HORIZON))


def test_render_places_lead_car_and_lanes():
    r = BEVRenderer(CameraConfig(hfov_deg=60.0, mount_height_m=CAM_H), corridor_half_width_m=1.2)
    img = r.render(_scene(), 720)
    assert img.shape == (720, 360, 3) and img.dtype == np.uint8
    cx, cy = r._px(0.0, 20.0 + 2.25, 720)  # middle of the car footprint (4.5 m long, visible face at 20 m)
    assert tuple(int(v) for v in img[cy, cx]) == THREAT  # in path, TTC 5 s: filled orange
    lx, ly = r._px(-1.75, 15.0, 720)  # left lane line at 15 m
    patch = img[ly - 2:ly + 3, lx - 2:lx + 3].reshape(-1, 3).astype(int)
    assert (np.abs(patch - (0, 230, 255)).sum(1) < 60).any()


def test_render_without_horizon_or_lanes():
    r = BEVRenderer(CameraConfig(), corridor_half_width_m=None)
    ctx = _scene(horizon_y=None)
    ctx.lanes = LaneInfo(image_width=W)
    assert r.render(ctx, 480).shape == (480, 360, 3)


def test_hud_appends_bev_panel():
    ctx = _scene()
    dec = DrivingDecision(LongAction.KEEP, LatAction.KEEP_LANE, 60.0)
    res = FrameResult(context=ctx, decision=dec, vlm_decision=None,
                      command=ControlCommand(0.2, 0.0, 0.0, 60.0, LongAction.KEEP, LatAction.KEEP_LANE))
    frame = np.zeros((H, W, 3), np.uint8)
    out = HUD(bev=BEVRenderer(CameraConfig())).draw(frame, res)
    assert out.shape == (H, W + 360, 3)


def test_pitch_error_range():
    near, far = pitch_error_range(40.0, 1.3, 0.5)
    assert near == pytest.approx(31.6, abs=0.2) and far == pytest.approx(54.8, abs=0.3)


def test_far_object_marked_on_top_edge():
    r = BEVRenderer(CameraConfig(hfov_deg=60.0, mount_height_m=CAM_H), corridor_half_width_m=None, range_m=60.0)
    ctx = _scene()
    ctx.detections = [Detection("car", 0.9, (630, 350, 650, 362), distance_m=80.0)]
    img = r.render(ctx, 720)
    _, top, _ = r._scale(720)
    assert tuple(int(v) for v in img[top + 8, 180]) == (80, 200, 80)  # marker at the centre of the top edge
