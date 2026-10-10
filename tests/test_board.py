import numpy as np

from adas_vla.perception.board import MEAN, STD, TensorInfo, _Boxes, decode_runs, parse_header, quant_lut

HEADER = """VLASTREAM 1
MODEL det yolo11s
IN images 0x408 0.00392156886 0 4 1 3 384 640
OUT scores 0x408 0.00367388339 0 3 1 80 5040
OUT boxes 0x408 2.58210492 0 3 1 4 5040
MODEL lane yolop_lane
IN images 0x416 7.26009603e-05 -29172 4 1 3 384 640
OUT lane_line_seg 0x416 1.52587891e-05 0 4 1 2 384 640""".splitlines()


def test_parse_header():
    models = parse_header(HEADER)
    assert models["det"]["graph"] == "yolo11s" and models["lane"]["graph"] == "yolop_lane"
    det_in, lane_in = models["det"]["in"][0], models["lane"]["in"][0]
    assert det_in.dims == [1, 3, 384, 640] and det_in.bits == 8 and det_in.offset == 0
    assert lane_in.bits == 16 and lane_in.offset == -29172
    assert [t.name for t in models["det"]["out"]] == ["scores", "boxes"]


def test_quant_lut_detector_is_identity():
    det_in = parse_header(HEADER)["det"]["in"][0]  # scale 1/255, offset 0: quantized input == pixel value
    lut = quant_lut(det_in)
    assert lut.shape == (3, 256) and all(np.array_equal(row, np.arange(256)) for row in lut)


def test_quant_lut_lane_matches_normalization():
    t = TensorInfo("images", 0x416, 7.26009603e-05, -29172, [1, 3, 384, 640])
    lut = quant_lut(t, MEAN, STD).astype(np.float64)
    real = (np.arange(256)[None] / 255.0 - np.array(MEAN)[:, None]) / np.array(STD)[:, None]
    deq = (lut + t.offset) * t.scale
    inside = (real > t.offset * t.scale) & (real < (65535 + t.offset) * t.scale)  # values not clipped
    assert np.abs(deq - real)[inside].max() <= t.scale * 0.51
    assert (np.diff(lut, axis=1) >= 0).all()


def test_decode_runs_roundtrip():
    rng = np.random.default_rng(0)
    mask = (rng.random((6, 10)) > 0.7).astype(np.uint8)
    flat, runs, cur, n = mask.ravel(), [], 0, 0
    for v in flat:  # encoder as in the server: alternate background / lane, starting with background
        if v != cur:
            runs.append(n)
            cur, n = v, 0
        n += 1
    runs.append(n)
    assert np.array_equal(decode_runs(np.array(runs, np.uint32), 6, 10), mask)


def test_tracker_boxes_view():
    b = _Boxes(np.array([[10, 20, 30, 60, 0.9, 2], [0, 0, 4, 4, 0.3, 0]], np.float32))
    assert np.allclose(b.xywh[0], [20, 40, 20, 40]) and len(b[b.conf > 0.5]) == 1 and b.cls[1] == 0
