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
    est = MotionEstimator(cfg.camera, dist_alpha=1.0, vel_alpha=1.0)
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
    est = MotionEstimator(cfg.camera, dist_alpha=1.0, vel_alpha=1.0)
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
