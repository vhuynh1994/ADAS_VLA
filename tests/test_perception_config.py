import cv2
import numpy as np
import pytest

from adas_vla.config import Config, load_config
from adas_vla.perception import LaneDetector, MotionEstimator, estimate_distance, focal_length_px
from adas_vla.types import Detection, LaneInfo


def synthetic_road(w=1280, h=720):
    img = np.full((h, w, 3), 60, np.uint8)
    cv2.line(img, (200, h), (600, int(0.62 * h)), (255, 255, 255), 8)
    cv2.line(img, (1080, h), (680, int(0.62 * h)), (255, 255, 255), 8)
    return img


def test_lane_detector_finds_both_lines_centered():
    lanes = LaneDetector()(synthetic_road())
    assert lanes.valid
    left, right = lanes.bounds_at(720)
    assert abs(left - 200) < 25 and abs(right - 1080) < 25
    assert abs(lanes.offset_norm) < 0.1


def test_lane_detector_no_lines():
    lanes = LaneDetector()(np.full((720, 1280, 3), 60, np.uint8))
    assert not lanes.valid and lanes.offset_norm is None


def test_distance_estimate_pinhole():
    f = focal_length_px(1280, 60.0)
    det = Detection("car", 0.9, (600, 400, 700, 400 + f * 1.5 / 20))  # 1.5 m tall car at 20 m
    assert estimate_distance(det, f) == pytest.approx(20.0)


def test_motion_estimator_ttc_for_approaching_object():
    cfg = Config()
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0)
    f = focal_length_px(1280, cfg.camera.hfov_deg)
    lanes = LaneInfo(image_width=1280)
    last = None
    for i, dist in enumerate([30.0, 29.0, 28.0]):  # closing at 10 m/s with dt = 0.1 s
        h_px = f * 1.5 / dist
        det = Detection("car", 0.9, (590, 600 - h_px, 690, 600), track_id=1)
        est.update([det], i * 0.1, lanes, 1280, 720)
        last = det
    assert last.in_ego_path and last.position == "ahead"
    assert last.closing_speed_mps == pytest.approx(10.0, rel=1e-3)
    assert last.ttc_s == pytest.approx(2.8, rel=1e-3)


def test_object_outside_corridor_not_in_path():
    est = MotionEstimator(Config().camera)
    det = Detection("car", 0.9, (0, 500, 120, 600))
    est.update([det], 0.0, LaneInfo(image_width=1280), 1280, 720)
    assert not det.in_ego_path and det.position == "front-left"


def test_config_defaults_and_overrides(tmp_path):
    cfg = load_config(overrides=["vlm.language=vi", "control.cruise_speed_kmh=80", "vlm.image_size=null"])
    assert cfg.vlm.language == "vi"
    assert cfg.control.cruise_speed_kmh == 80
    assert cfg.vlm.image_size is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("vlm:\n  nope: 1\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_pc_fast_profile_loads():
    from adas_vla.config import DEFAULT_CONFIG_PATH

    cfg = load_config(DEFAULT_CONFIG_PATH.parent / "pc_fast.yaml")
    assert cfg.vlm.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert cfg.vlm.quantization == "none"
    assert cfg.safety.aeb_ttc_s == Config().safety.aeb_ttc_s


def test_oncoming_vehicle_flagged_and_not_described_as_threat():
    cfg = Config()
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0)
    f = focal_length_px(1280, cfg.camera.hfov_deg)
    lanes = LaneInfo(image_width=1280)
    ego_mps = 60 / 3.6
    last = None
    for i, dist in enumerate([80.0, 77.0, 74.0]):  # closing at 30 m/s while ego drives 16.7 m/s
        h_px = f * 1.5 / dist
        det = Detection("car", 0.9, (100, 450 - h_px, 140, 450), track_id=7)
        est.update([det], i * 0.1, lanes, 1280, 720, ego_mps)
        last = det
    assert last.oncoming and not last.in_ego_path
    text = last.describe()
    assert "oncoming" in text and "TTC" not in text


def test_event_trigger_fires_on_scene_change():
    from adas_vla.pipeline import ADASPipeline
    from adas_vla.types import EgoState, SceneContext

    pipe = ADASPipeline.__new__(ADASPipeline)  # no models needed for the trigger logic
    pipe.cfg = Config()
    pipe.cfg.vlm.every_n_frames, pipe.cfg.vlm.min_interval_frames = 30, 5
    pipe._last_vlm_frame, pipe._last_signature = None, None

    def ctx(i, dets=()):
        return SceneContext(i, i / 30, 1280, 720, EgoState(60), list(dets), LaneInfo(image_width=1280))

    calls = []
    lead = Detection("car", 0.9, (600, 400, 700, 480), track_id=3, distance_m=25, in_ego_path=True)
    for i in range(60):
        c = ctx(i, [lead] if i >= 12 else [])
        if pipe.should_call_vlm(c):
            pipe._last_vlm_frame = i
            calls.append(i)
    assert calls == [0, 12, 42]  # start, new lead vehicle, then the regular interval


def test_lane_change_detection_on_smoothed_offsets():
    from adas_vla.datasets.comma2k19 import detect_lane_changes, lateral_label
    from adas_vla.types import LatAction

    # 20 fps, sampled every 4 frames; drift left, 2 samples lost while crossing, smoothed swing to the new lane
    trace = [-0.1, -0.2, -0.3, None, None, -0.3, -0.5, 0.1, 0.2, 0.1, 0.0, -0.1]
    offsets = [(i * 4, o) for i, o in enumerate(trace)]
    events = detect_lane_changes(offsets)
    assert events == [(28, LatAction.CHANGE_LEFT)]
    assert lateral_label(0, events) is LatAction.CHANGE_LEFT  # within 3 s before the crossing
    assert lateral_label(200, events) is LatAction.KEEP_LANE
    # normal in-lane wobble is not a lane change
    wobble = [(i * 4, o) for i, o in enumerate([-0.3, -0.1, 0.1, 0.2, 0.0, -0.2, -0.3, -0.1])]
    assert detect_lane_changes(wobble) == []
    # mirrored: right lane change
    right = [(i * 4, o) for i, o in enumerate([0.2, 0.4, 0.5, -0.2, -0.3])]
    assert detect_lane_changes(right)[0][1] is LatAction.CHANGE_RIGHT


def test_ego_hood_is_not_a_vehicle():
    from adas_vla.perception.detector import is_ego_hood

    assert is_ego_hood((3, 625, 1161, 860), 1164, 874)  # comma2k19 hood seen as a 'car'
    assert not is_ego_hood((421, 390, 462, 413), 1164, 874)  # distant car
    assert not is_ego_hood((829, 338, 1164, 540), 1164, 874)  # large close car in the next lane
    assert not is_ego_hood((300, 300, 1000, 870), 1164, 874)  # truck right in front: tall box, top not low
    # Australian dashcam: hood + dashboard + road seen as one full-width 'car' (phantom AEB at "2 m")
    assert is_ego_hood((0, 263, 1280, 700), 1280, 720)
    assert not is_ego_hood((40, 200, 1240, 500), 1280, 720)  # bus crossing ahead: wide, but on the road


def test_object_on_dashboard_is_not_close():
    """Australian dashcam regression: a small ornament on the dashboard, detected as a 'person' touching the
    frame bottom, got the ground-plane distance of the bottom row (4 m) and triggered AEB + VRU braking."""
    f = focal_length_px(1280, 60.0)
    ornament = Detection("person", 0.5, (831, 663, 896, 719))
    dist = estimate_distance(ornament, f, frame_height=720, cam_height_m=1.3)
    assert dist == pytest.approx(estimate_distance(ornament, f)) and dist > 30


def test_nexar_sample_times():
    from adas_vla.datasets.nexar import sample_times
    from adas_vla.types import LongAction

    times = sample_times(alert=18.6, event=19.5)
    keeps = [t for t, a in times if a is LongAction.KEEP]
    brakes = [t for t, a in times if a is LongAction.BRAKE]
    assert keeps == [18.6 - 7.0, 18.6 - 4.0]
    assert all(18.6 <= t < 19.5 for t in brakes) and len(brakes) == 2
    # very short alert window: a single BRAKE sample, never at or after the event
    short = sample_times(alert=5.0, event=5.3)
    assert [a for _, a in short].count(LongAction.BRAKE) == 1 and all(t < 5.3 for t, _ in short)
    assert all(t >= 0.5 for t, _ in short)  # no KEEP sample before the video starts


def test_clipped_box_uses_width_and_ground_plane():
    from adas_vla.perception.geometry import ground_distance

    f = focal_length_px(1280, 60.0)
    truck = Detection("truck", 0.9, (300, 150, 1000, 715))  # bottom edge cut by the frame: only 565 px tall
    assert estimate_distance(truck, f) == pytest.approx(f * 3.0 / 565)  # the height alone says ~5.9 m
    close = estimate_distance(truck, f, frame_height=720, cam_height_m=1.3)
    assert close == pytest.approx(min(f * 2.5 / 700, ground_distance(715, f, 720, 1.3))) and close < 5
    # the same box with its top below the horizon would be a 0.8 m high "truck" at 4 m: not a clipped truck
    low = Detection("truck", 0.9, (300, 500, 1000, 715))
    assert estimate_distance(low, f, 720, 1.3) == pytest.approx(estimate_distance(low, f))
    unclipped = Detection("truck", 0.9, (300, 500, 1000, 690))
    assert estimate_distance(unclipped, f, 720, 1.3) == pytest.approx(estimate_distance(unclipped, f))
    # applied by the MotionEstimator: the same truck in the corridor is a lead a few metres away
    est = MotionEstimator(Config().camera)
    est.update([truck], 0.0, LaneInfo(image_width=1280), 1280, 720)
    assert truck.in_ego_path and truck.distance_m < 5


def test_cut_in_detected_then_becomes_lead():
    from adas_vla.types import EgoState, SceneContext

    cfg = Config()
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0)
    h_px = focal_length_px(1280, cfg.camera.hfov_deg) * 1.5 / 20  # car 20 m ahead
    lanes = LaneInfo(image_width=1280)
    seen = []
    for k, x1 in enumerate([1000, 960, 920, 880, 840]):  # right lane, moving left into our corridor (10 Hz)
        det = Detection("car", 0.9, (x1, 600 - h_px, x1 + 100, 600), track_id=5)
        est.update([det], k * 0.1, lanes, 1280, 720)
        seen.append((det.cutting_in, det.in_ego_path))
    assert seen[:3] == [(False, False)] * 3  # not enough history to judge the lateral motion
    assert seen[3] == (True, False)  # closing on the corridor fast, about to overlap it
    assert seen[4] == (False, True)  # inside the corridor: simply the lead now
    det = Detection("car", 0.9, (880, 600 - h_px, 980, 600), track_id=5, distance_m=20.0, cutting_in=True,
                    position="front-right", closing_speed_mps=2.0)
    ctx = SceneContext(0, 0.0, 1280, 720, EgoState(50), [det], lanes)
    assert ctx.lead_object() is det
    assert "CUTTING IN from the right" in det.describe() and "closing in" in det.describe()


def test_overtaking_car_pulling_away_is_not_a_cut_in():
    """Highway regression (highway_traffic.mp4, t = 30.5 s): a car overtaking in the next lane on a curve drifts
    toward the corridor in the image while pulling away from us; it must not become a cut-in lead (false FCW)."""
    cfg = Config()
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0)
    f = focal_length_px(1280, cfg.camera.hfov_deg)
    for k, x1 in enumerate([1000, 960, 920, 880, 840]):
        dist = 11.0 + 0.2 * k  # 2 m/s faster than us
        h_px = f * 1.5 / dist
        det = Detection("car", 0.9, (x1, 600 - h_px, x1 + 100, 600), track_id=6)
        est.update([det], k * 0.1, LaneInfo(image_width=1280), 1280, 720)
        assert not det.cutting_in


def test_adjacent_car_wobble_is_not_a_cut_in():
    cfg = Config()
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0)
    h_px = focal_length_px(1280, cfg.camera.hfov_deg) * 1.5 / 20
    for k in range(12):
        x1 = 940 + (6 if k % 2 else -6)  # box jitter of an adjacent car driving straight
        det = Detection("car", 0.9, (x1, 600 - h_px, x1 + 100, 600), track_id=9)
        est.update([det], k * 0.1, LaneInfo(image_width=1280), 1280, 720)
        assert not det.cutting_in and not det.in_ego_path


def test_frame_history_returns_frame_gap_earlier():
    from adas_vla.sources import FrameHistory

    hist = FrameHistory(0.5)
    for i in range(11):
        hist.push(i * 0.1, np.full((2, 2), i, np.uint8))
    assert hist.before(0.3) is None
    assert int(hist.before(1.0)[0, 0]) == 5  # the frame at t = 0.5
    assert int(hist.before(0.75)[0, 0]) == 2  # newest frame at least 0.5 s old
    hist.reset()
    assert hist.before(1.0) is None
    off = FrameHistory(0.0)
    off.push(0.0, np.zeros((2, 2), np.uint8))
    assert off.before(1.0) is None  # disabled: stores nothing


def test_hazard_cue_from_perception_summary():
    from adas_vla.types import EgoState, SceneContext, context_has_hazard_cue

    lanes = LaneInfo(image_width=1280)

    def summary(**kw):
        det = Detection("car", 0.9, (600, 400, 700, 480), **kw)
        return SceneContext(0, 0.0, 1280, 720, EgoState(50), [det], lanes).summary_text()

    assert context_has_hazard_cue(summary(distance_m=25, in_ego_path=True, closing_speed_mps=3.0))
    assert not context_has_hazard_cue(summary(distance_m=25, in_ego_path=True, closing_speed_mps=-3.0))  # pulling away
    assert not context_has_hazard_cue(summary(distance_m=60, in_ego_path=True))  # too far
    assert not context_has_hazard_cue(summary(distance_m=10, in_ego_path=False))  # not in our lane
    assert context_has_hazard_cue(summary(distance_m=20, cutting_in=True, position="front-left"))
    assert context_has_hazard_cue("- traffic light (red), ahead, ~30 m")
    assert not context_has_hazard_cue("Lane: lane lines not detected.\nDetected objects: none.")
