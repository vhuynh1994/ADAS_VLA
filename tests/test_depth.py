"""Depth-Anything distance refinement (perception/depth.py): CPU-only, synthetic maps, no network."""

import numpy as np
import pytest

from adas_vla.config import Config, load_config
from adas_vla.perception import DepthFrame, MotionEstimator
from adas_vla.perception.depth import (DepthAligner, box_stat, depth_frame, fit_scale_shift, ground_anchors,
                                       preprocess)
from adas_vla.perception.geometry import estimate_distance, focal_length_px, ground_distance
from adas_vla.types import Detection, LaneInfo

W, H = 1280, 720
S, T = 0.8, 0.02  # the unknown affine map of a relative-depth model: 1 / d = S * value + T


def value_for(dist: float) -> float:
    """Map value a perfect relative-depth model outputs for a point `dist` metres away."""
    return (1.0 / dist - T) / S


def road_anchors(cfg: Config) -> list[tuple[float, float]]:
    f = focal_length_px(W, cfg.camera.hfov_deg)
    pairs = []
    for ry in np.linspace(0.62, 0.88, 8):
        d = ground_distance(ry * H, f, H, cfg.camera.mount_height_m)
        pairs.append((value_for(d), 1.0 / d))
    return pairs


def car_box(dist: float, f: float, x_center: float = 640, bottom: float = 600) -> tuple[float, float, float, float]:
    h_px, w_px = f * 1.5 / dist, f * 1.8 / dist
    return (x_center - w_px / 2, bottom - h_px, x_center + w_px / 2, bottom)


def test_fit_scale_shift_recovers_affine_map_with_outliers():
    rng = np.random.default_rng(0)
    pairs = [(value_for(d) * (1 + rng.normal(0, 0.01)), 1.0 / d) for d in np.linspace(6, 60, 20)]
    pairs += [(value_for(8.0), 1.0 / 40.0), (value_for(50.0), 1.0 / 7.0)]  # two gross outliers
    fit = fit_scale_shift(pairs, min_n=6, max_rel_rmse=0.25)
    assert fit is not None
    assert fit.scale == pytest.approx(S, rel=0.05) and fit.shift == pytest.approx(T, abs=0.005)
    assert fit.distance(value_for(20.0)) == pytest.approx(20.0, rel=0.05)
    assert fit.n >= 18
    assert fit.distance(None) is None
    assert fit.distance(value_for(1e6)) is None  # beyond the horizon: no distance rather than a huge one


def test_fit_scale_shift_rejects_degenerate_inputs():
    assert fit_scale_shift([(1.0, 0.1)] * 10) is None  # no spread in the map values
    assert fit_scale_shift([(value_for(d), 1.0 / d) for d in (10, 20, 30)], min_n=6) is None  # too few anchors
    # a map that gets smaller for closer points is not a disparity
    assert fit_scale_shift([(-value_for(d), 1.0 / d) for d in np.linspace(5, 50, 10)]) is None
    # anchors that do not fit one affine map (random) are rejected by the residual check
    rng = np.random.default_rng(1)
    assert fit_scale_shift([(rng.uniform(0, 1), rng.uniform(0.01, 0.2)) for _ in range(30)]) is None


def test_box_stat_reads_the_object_not_its_outline():
    dmap = np.full((36, 64), 1.0, np.float32)  # map at 1/20 of a 1280x720 frame
    dmap[17:29, 23:34] = 5.0  # object body inside the box 400..720 x 300..600 (map 20..36 x 15..30); rim = 1.0
    assert box_stat(dmap, (400, 300, 720, 600), W, H) == pytest.approx(5.0)
    graded = np.zeros((36, 64), np.float32)
    graded[17:29, 23:34] = np.linspace(1, 9, 11)[None, :]
    hi = box_stat(graded, (400, 300, 720, 600), W, H, percentile=80, higher_is_closer=True)
    lo = box_stat(graded, (400, 300, 720, 600), W, H, percentile=80, higher_is_closer=False)
    assert hi > 5 > lo  # "toward the camera" means high disparity, low metric depth
    assert box_stat(dmap, (0, 0, 3, 3), W, H) == pytest.approx(1.0)  # tiny box: whole-box fallback
    assert box_stat(dmap, (1279, 719, 1280, 720), W, H) is not None


def test_ground_anchors_follow_the_flat_road_and_skip_boxes():
    cfg = Config()
    f = focal_length_px(W, cfg.camera.hfov_deg)
    dmap = np.zeros((72, 128), np.float32)
    for my in range(72):  # a perfect map of an empty flat road
        d = ground_distance((my + 0.5) * 10, f, H, cfg.camera.mount_height_m)
        dmap[my, :] = value_for(d) if d else 0.0
    pairs = ground_anchors(dmap, W, H, [], f, cfg.camera.mount_height_m)
    assert len(pairs) == 40  # 8 rows x 5 columns
    fit = fit_scale_shift(pairs)
    assert fit is not None and fit.scale == pytest.approx(S, rel=0.05) and fit.shift == pytest.approx(T, abs=0.005)
    fewer = ground_anchors(dmap, W, H, [(560, 500, 720, 720)], f, cfg.camera.mount_height_m)
    assert 0 < len(fewer) < len(pairs)  # points under a vehicle are not road


def test_depth_replaces_the_pinhole_distance_once_aligned():
    cfg = Config()
    cfg.perception.depth.anchors = ["ground"]
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0, depth=cfg.perception.depth)
    f = focal_length_px(W, cfg.camera.hfov_deg)
    det = Detection("car", 0.9, car_box(20.0, f), track_id=1)  # the box says 20 m ...
    frame = DepthFrame("relative_disparity", stats=[value_for(15.0)], anchors=road_anchors(cfg))  # ... the map 15 m
    est.update([det], 0.0, LaneInfo(image_width=W), W, H, depth=frame)
    assert det.distance_source == "depth" and det.distance_m == pytest.approx(15.0, rel=0.02)
    # the alignment is per video: a reset forgets it
    est.reset()
    est.update([det], 0.0, LaneInfo(image_width=W), W, H, depth=DepthFrame("relative_disparity", [value_for(15.0)]))
    assert det.distance_source == "pinhole" and det.distance_m == pytest.approx(20.0, rel=1e-3)


def test_depth_mode_min_is_never_farther_than_the_pinhole_estimate():
    cfg = Config()
    cfg.perception.depth.mode = "min"
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0, depth=cfg.perception.depth)
    f = focal_length_px(W, cfg.camera.hfov_deg)
    det = Detection("car", 0.9, car_box(20.0, f), track_id=1)
    est.update([det], 0.0, LaneInfo(image_width=W), W, H,
               depth=DepthFrame("relative_disparity", [value_for(30.0)], road_anchors(cfg)))
    assert det.distance_m == pytest.approx(20.0, rel=1e-3)  # depth farther: pinhole kept
    est.update([det], 0.1, LaneInfo(image_width=W), W, H,
               depth=DepthFrame("relative_disparity", [value_for(15.0)], road_anchors(cfg)))
    assert det.distance_source == "depth" and det.distance_m == pytest.approx(15.0, rel=0.02)


def test_clipped_box_is_capped_by_its_pinhole_upper_bound():
    cfg = Config()
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0, depth=cfg.perception.depth)
    f = focal_length_px(W, cfg.camera.hfov_deg)
    box = (400, 300, 880, 720)  # cut by the bottom edge: a very close vehicle
    bound = estimate_distance(Detection("car", 0.9, box), f, H, cfg.camera.mount_height_m)
    assert 3.5 < bound < 4.5
    det = Detection("car", 0.9, box, track_id=2)
    est.update([det], 0.0, LaneInfo(image_width=W), W, H,
               depth=DepthFrame("relative_disparity", [value_for(6.0)], road_anchors(cfg)))
    assert det.distance_m == pytest.approx(bound)  # never farther than the clipped-box bound
    det2 = Detection("car", 0.9, box, track_id=3)
    est.update([det2], 0.1, LaneInfo(image_width=W), W, H,
               depth=DepthFrame("relative_disparity", [value_for(3.0)], road_anchors(cfg)))
    assert det2.distance_source == "depth" and det2.distance_m == pytest.approx(3.0, rel=0.02)


def test_depth_falls_back_to_pinhole_without_a_reliable_alignment():
    cfg = Config()
    f = focal_length_px(W, cfg.camera.hfov_deg)
    lanes = LaneInfo(image_width=W)
    few = road_anchors(cfg)[:3]  # fewer than min_anchors
    for mode, anchors, stat in (("replace", few, value_for(15.0)), ("off", road_anchors(cfg), value_for(15.0)),
                                ("replace", road_anchors(cfg), None)):
        cfg.perception.depth.mode = mode
        est = MotionEstimator(cfg.camera, dist_tau_s=0.0, depth=cfg.perception.depth)
        det = Detection("car", 0.9, car_box(20.0, f), track_id=1)
        est.update([det], 0.0, lanes, W, H, depth=DepthFrame("relative_disparity", [stat], anchors))
        assert det.distance_source == "pinhole" and det.distance_m == pytest.approx(20.0, rel=1e-3), mode
    # no depth config at all (the default MotionEstimator) ignores depth frames
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0)
    det = Detection("car", 0.9, car_box(20.0, f), track_id=1)
    est.update([det], 0.0, lanes, W, H, depth=DepthFrame("relative_disparity", [value_for(15.0)], road_anchors(cfg)))
    assert det.distance_source == "pinhole" and det.distance_m == pytest.approx(20.0, rel=1e-3)


def test_box_anchors_align_the_map_when_the_road_is_not_visible():
    cfg = Config()
    cfg.perception.depth.anchors = ["boxes"]
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0, depth=cfg.perception.depth)
    f = focal_length_px(W, cfg.camera.hfov_deg)
    dists = [8.0, 12.0, 18.0, 25.0, 35.0, 50.0, 70.0]
    dets = [Detection("car", 0.9, car_box(d, f, x_center=200 + 150 * i, bottom=400 + 2 * i), track_id=10 + i)
            for i, d in enumerate(dists)]
    target = Detection("truck", 0.9, car_box(30.0, f, x_center=640, bottom=650), track_id=99)
    stats = [value_for(d) for d in dists] + [value_for(22.0)]  # the map says the truck is at 22 m, not 30
    est.update(dets + [target], 0.0, LaneInfo(image_width=W), W, H,
               depth=DepthFrame("relative_disparity", stats, anchors=[]))
    assert target.distance_source == "depth" and target.distance_m == pytest.approx(22.0, rel=0.03)


def test_metric_depth_is_used_directly():
    cfg = Config()
    cfg.perception.depth.output_kind = "metric_depth"
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0, depth=cfg.perception.depth)
    f = focal_length_px(W, cfg.camera.hfov_deg)
    det = Detection("car", 0.9, car_box(20.0, f), track_id=1)
    est.update([det], 0.0, LaneInfo(image_width=W), W, H, depth=DepthFrame("metric_depth", [12.5]))
    assert det.distance_source == "depth" and det.distance_m == pytest.approx(12.5)
    est.update([det], 0.1, LaneInfo(image_width=W), W, H, depth=DepthFrame("metric_depth", [0.0]))
    assert det.distance_source == "pinhole"


def test_ttc_from_depth_distances_through_the_motion_estimator():
    cfg = Config()
    cfg.perception.depth.anchors = ["ground"]
    est = MotionEstimator(cfg.camera, dist_tau_s=0.0, depth=cfg.perception.depth)
    f = focal_length_px(W, cfg.camera.hfov_deg)
    last = None
    for i, dist in enumerate([30.0, 29.0, 28.0]):  # the map sees the car closing at 10 m/s; the box says 40 m
        det = Detection("car", 0.9, car_box(40.0, f), track_id=5)
        est.update([det], i * 0.1, LaneInfo(image_width=W), W, H,
                   depth=DepthFrame("relative_disparity", [value_for(dist)], road_anchors(cfg)))
        last = det
    assert last.distance_source == "depth" and last.distance_m == pytest.approx(28.0, rel=0.02)
    assert last.closing_speed_mps == pytest.approx(10.0, rel=0.05)
    assert last.ttc_s == pytest.approx(2.8, rel=0.05)


def test_aligner_keeps_then_forgets_the_alignment():
    cfg = Config()
    aligner = DepthAligner(cfg.perception.depth)
    first = aligner.update(road_anchors(cfg), 0.0)
    assert first is not None
    assert aligner.update([], 0.5) is first  # no fit this frame: the last good one stays within align_max_age_s
    assert aligner.update([], 2.0) is None  # too old
    cfg.perception.depth.align_tau_s = 1.0
    aligner.update(road_anchors(cfg), 10.0)
    doubled = [(v, 2 * y) for v, y in road_anchors(cfg)]  # a sudden change is only followed gradually
    smoothed = aligner.update(doubled, 10.1)
    assert S < smoothed.scale < 2 * S


def test_depth_frame_round_trip_and_subset():
    frame = DepthFrame("relative_disparity", [0.5, None, 0.25], [(0.1, 0.05), (0.2, 0.1)])
    back = DepthFrame.from_dict(frame.to_dict())
    assert back == frame
    sub = frame.subset([2, 0])
    assert sub.stats == [0.25, 0.5] and sub.anchors == frame.anchors and sub.kind == frame.kind


def test_depth_frame_builder():
    cfg = Config()
    dmap = np.full((36, 64), 0.1, np.float32)
    dets = [Detection("car", 0.9, (400, 300, 720, 600)), Detection("person", 0.9, (900, 400, 940, 520))]
    frame = depth_frame(dmap, W, H, dets, cfg.perception.depth, cfg.camera)
    assert frame.kind == "relative_disparity" and frame.stats == [pytest.approx(0.1)] * 2 and frame.anchors
    cfg.perception.depth.output_kind = "metric_depth"
    metric = depth_frame(dmap, W, H, dets, cfg.perception.depth, cfg.camera)
    assert metric.kind == "metric_depth" and metric.anchors == []


def test_preprocess_shape_and_normalisation():
    frame = np.full((720, 1280, 3), 255, np.uint8)
    x = preprocess(frame, 644, 364)
    assert x.shape == (1, 3, 364, 644) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229)
    assert preprocess(frame, 64, 32, normalize=False).max() == pytest.approx(1.0)


def test_depth_config_defaults_and_overrides():
    cfg = load_config()
    assert cfg.perception.depth.enabled is False and cfg.perception.depth.mode == "replace"
    cfg = load_config(overrides=["perception.depth.mode=min", "perception.depth.input_size=[518,518]",
                                 "perception.depth.backend=onnx"])
    assert cfg.perception.depth.mode == "min" and cfg.perception.depth.input_size == [518, 518]
    assert cfg.perception.depth.backend == "onnx"
    with pytest.raises(ValueError):
        load_config(overrides=["perception.depth.nope=1"])
