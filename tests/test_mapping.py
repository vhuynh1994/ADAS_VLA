"""Route lane map helpers: geodesy round trips, headings, placing road points, lane-marking peaks."""

import math

import numpy as np
import pytest

from adas_vla.mapping import (LocalFrame, camera_heading, ecef_to_lla, find_peaks, heading_of, lateral_profile,
                              lla_to_ecef, offset_profile, path_offsets, quat_to_rot, to_world, wrap)


def test_ecef_lla_round_trip():
    for lat, lon, alt in [(37.33, -121.89, 30.0), (-33.9, 151.2, 5.0), (60.0, 10.0, 500.0)]:
        back = ecef_to_lla(lla_to_ecef(lat, lon, alt))[0]
        assert back == pytest.approx([lat, lon, alt], abs=1e-6)


def test_local_frame_east_and_north():
    ref = lla_to_ecef(37.33, -121.89, 0.0)
    frame = LocalFrame(ref)
    north = frame.to_enu(lla_to_ecef(37.33 + 0.001, -121.89, 0.0))[0]  # ~111 m north
    east = frame.to_enu(lla_to_ecef(37.33, -121.89 + 0.001, 0.0))[0]  # ~88 m east
    assert north[1] == pytest.approx(111.0, abs=1.0) and abs(north[0]) < 0.01
    assert east[0] == pytest.approx(88.6, abs=1.0) and abs(east[1]) < 0.5
    assert frame.to_lla(north[:2])[0][:2] == pytest.approx([37.331, -121.89], abs=1e-6)


def test_heading_from_quaternion_matches_rotation():
    frame = LocalFrame(lla_to_ecef(37.33, -121.89, 0.0))
    # device x axis pointing east, z down: build the device->ECEF rotation, convert it to a quaternion
    east, north, up = frame.R
    rot = np.stack([east, -north, -up], 1)  # columns: device x, y, z in ECEF (y right of east-facing = south)
    w = math.sqrt(1 + np.trace(rot)) / 2
    q = np.array([w, (rot[2, 1] - rot[1, 2]) / (4 * w), (rot[0, 2] - rot[2, 0]) / (4 * w), (rot[1, 0] - rot[0, 1]) / (4 * w)])
    assert quat_to_rot(q)[0] == pytest.approx(rot, abs=1e-9)
    assert camera_heading(q, frame)[0] == pytest.approx(math.pi / 2)  # east
    assert heading_of(np.array([0.0, -1.0, 0.0]))[0] == pytest.approx(math.pi)  # south
    assert wrap(np.array([3 * math.pi]))[0] == pytest.approx(-math.pi)


def test_to_world_heading_east():
    pts = to_world(np.array([1.0]), np.array([10.0]), np.array([100.0, 50.0]), math.pi / 2)
    assert pts[0] == pytest.approx([110.0, 49.0])  # 10 m ahead = east, 1 m right = south


def test_profile_finds_lane_lines_of_a_straight_road():
    rng = np.random.default_rng(0)
    along = rng.uniform(-60, 60, 6000)
    lines = rng.choice([-5.55, -1.85, 1.85, 5.55], 6000)  # lane lines 3.7 m apart, road heading north-east
    across = lines + rng.normal(0, 0.08, 6000)
    h = math.radians(45)
    pts = np.stack([along * math.sin(h) + across * math.cos(h), along * math.cos(h) - across * math.sin(h)], 1)
    centres, counts = lateral_profile(pts, np.zeros(2), h)
    peaks = find_peaks(centres, counts)
    assert [round(p[0], 1) for p in peaks] == pytest.approx([-5.55, -1.85, 1.85, 5.55], abs=0.1)
    assert all(0.1 < p[1] < 0.4 for p in peaks)  # FWHM of a 0.08 m sigma ~ 0.19 m (+ smoothing)


def test_path_offsets_follow_a_curve():
    # quarter circle of radius 200 m, travel counter-clockwise; marking points 1.85 m right of it (outside)
    a = np.linspace(0, math.pi / 2, 400)
    traj = np.stack([200 * np.cos(a), 200 * np.sin(a)], 1)
    heads = np.arctan2(-np.sin(a), np.cos(a))  # tangent of (cos a, sin a) moving with a: (-sin a, cos a) -> heading
    pts = np.stack([(200 + 1.85) * np.cos(a), (200 + 1.85) * np.sin(a)], 1)
    idx, off, along = path_offsets(pts, traj, heads)
    assert np.allclose(off, 1.85, atol=0.02) and np.allclose(along, 0.0, atol=0.02)
    _, _, along2 = path_offsets(traj[100:101] + [[-0.3 * math.sin(a[100]), 0.3 * math.cos(a[100])]], traj, heads)
    assert along2[0] == pytest.approx(0.3, abs=0.01)  # 0.3 m further along the path than sample 100
    centres, counts = offset_profile(off)
    assert find_peaks(centres, counts)[0][0] == pytest.approx(1.85, abs=0.05)
