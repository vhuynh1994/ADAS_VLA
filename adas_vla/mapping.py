"""Route lane map (map step 2): the BEV lane-marking points of every frame placed in a local east-north frame with
the drive's global pose, accumulated over the route.

Pose as in comma2k19 `global_pose/`: positions in ECEF (m), orientations as quaternions rotating the device frame
(x forward, y right, z down) to ECEF. Only the camera heading enters the map: the pitch comes from the lane
vanishing point as in the BEV view (bev.py), so a road point X m right and Z m ahead of the camera lands at
p_en + Z * forward + X * right, with forward / right horizontal unit vectors.

Quality without external data: on a straight stretch the accumulated markings, seen across the road, must be
narrow peaks (a 15 cm painted line blurred by pose and projection errors) spaced by one lane width (3.6-3.7 m on
US highways). `lateral_profile` + `find_peaks` measure both.
"""

from __future__ import annotations

import math

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def lla_to_ecef(lat_deg: float, lon_deg: float, alt: float = 0.0) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    return np.array([(n + alt) * math.cos(lat) * math.cos(lon), (n + alt) * math.cos(lat) * math.sin(lon),
                     (n * (1 - WGS84_E2) + alt) * math.sin(lat)])


def ecef_to_lla(xyz: np.ndarray) -> np.ndarray:
    """(N, 3) ECEF -> (N, 3) latitude (deg), longitude (deg), height (m); fixed-point iteration, mm accuracy."""
    xyz = np.atleast_2d(np.asarray(xyz, float))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.stack([np.degrees(lat), np.degrees(lon), h], 1)


class LocalFrame:
    """East-north-up frame tangent to the earth at `ref_ecef` (the first pose of a route)."""

    def __init__(self, ref_ecef: np.ndarray):
        self.ref = np.asarray(ref_ecef, float)
        lat, lon, _ = ecef_to_lla(self.ref)[0]
        la, lo = math.radians(lat), math.radians(lon)
        self.lat0, self.lon0 = lat, lon
        self.R = np.array([[-math.sin(lo), math.cos(lo), 0.0],  # rows: east, north, up in ECEF
                           [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                           [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])

    def to_enu(self, ecef: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(ecef) - self.ref) @ self.R.T

    def vec_to_enu(self, v_ecef: np.ndarray) -> np.ndarray:
        return np.atleast_2d(v_ecef) @ self.R.T

    def to_lla(self, enu: np.ndarray) -> np.ndarray:
        enu = np.atleast_2d(enu)
        if enu.shape[1] == 2:
            enu = np.hstack([enu, np.zeros((len(enu), 1))])
        return ecef_to_lla(self.ref + enu @ self.R)


def quat_to_rot(q: np.ndarray, scalar_first: bool = True) -> np.ndarray:
    """(N, 4) unit quaternions -> (N, 3, 3) rotation matrices."""
    q = np.atleast_2d(np.asarray(q, float))
    w, x, y, z = (q[:, 0], q[:, 1], q[:, 2], q[:, 3]) if scalar_first else (q[:, 3], q[:, 0], q[:, 1], q[:, 2])
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def heading_of(v_enu: np.ndarray) -> np.ndarray:
    """Heading (rad, clockwise from north) of horizontal vectors given in ENU."""
    v = np.atleast_2d(v_enu)
    return np.arctan2(v[:, 0], v[:, 1])


def camera_heading(q: np.ndarray, frame: LocalFrame, scalar_first: bool = True) -> np.ndarray:
    """Heading of the device x axis (the camera's forward direction) for each orientation quaternion."""
    forward_ecef = quat_to_rot(q, scalar_first)[:, :, 0]
    return heading_of(frame.vec_to_enu(forward_ecef))


def wrap(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def to_world(x_right: np.ndarray, z_ahead: np.ndarray, p_en: np.ndarray, heading: float) -> np.ndarray:
    """Road points (X right, Z ahead of the camera, m) -> (N, 2) east-north for a camera at p_en facing `heading`."""
    s, c = math.sin(heading), math.cos(heading)
    east = p_en[0] + z_ahead * s + x_right * c
    north = p_en[1] + z_ahead * c - x_right * s
    return np.stack([east, north], 1)


def lateral_profile(points_en: np.ndarray, center_en: np.ndarray, heading: float, length_m: float = 120.0,
                    half_width_m: float = 12.0, bin_m: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of the points' signed offsets across the road (+ = right of travel) within a window of the route
    `length_m` long centred on `center_en`; returns (bin centres, counts)."""
    d = points_en - center_en
    s, c = math.sin(heading), math.cos(heading)
    along = d[:, 0] * s + d[:, 1] * c
    across = d[:, 0] * c - d[:, 1] * s
    keep = (np.abs(along) <= length_m / 2) & (np.abs(across) <= half_width_m)
    edges = np.arange(-half_width_m, half_width_m + bin_m / 2, bin_m)
    counts, _ = np.histogram(across[keep], edges)
    return (edges[:-1] + edges[1:]) / 2, counts


def find_peaks(centres: np.ndarray, counts: np.ndarray, smooth_m: float = 0.1, min_sep_m: float = 2.0,
               min_frac: float = 0.15) -> list[tuple[float, float, float]]:
    """Lane-marking peaks of a lateral profile: [(offset m, full width at half maximum m, height)], sorted by
    offset. A peak must reach `min_frac` of the highest one and be `min_sep_m` from a higher peak."""
    if counts.sum() == 0:
        return []
    bin_m = centres[1] - centres[0]
    k = max(1, round(smooth_m / bin_m))
    y = np.convolve(counts.astype(float), np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    order = np.argsort(y)[::-1]
    peaks: list[int] = []
    for i in order:
        if y[i] < min_frac * y[order[0]]:
            break
        if all(abs(centres[i] - centres[j]) >= min_sep_m for j in peaks):
            peaks.append(int(i))
    out = []
    for i in sorted(peaks):
        half = y[i] / 2
        lo = i
        while lo > 0 and y[lo] > half:
            lo -= 1
        hi = i
        while hi < len(y) - 1 and y[hi] > half:
            hi += 1
        seg = slice(lo, hi + 1)  # centroid of the part above half maximum: robust to flat tops
        out.append((float((centres[seg] * y[seg]).sum() / y[seg].sum()), float((hi - lo) * bin_m), float(y[i])))
    return out


def path_offsets(points_en: np.ndarray, traj_en: np.ndarray, heads: np.ndarray,
                 chunk: int = 20000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curvilinear coordinates of points along a driven path: (index of the nearest path sample, signed offset to
    the right of the direction of travel there, offset along the travel direction from that sample). Follows
    curves, unlike a straight window; the along-path position of a point is dist[index] + along."""
    idx = np.empty(len(points_en), np.int64)
    for a in range(0, len(points_en), chunk):
        d2 = ((points_en[a:a + chunk, None, :] - traj_en[None, :, :]) ** 2).sum(-1)
        idx[a:a + chunk] = d2.argmin(1)
    d = points_en - traj_en[idx]
    s, c = np.sin(heads[idx]), np.cos(heads[idx])
    return idx, d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c


def offset_profile(offsets: np.ndarray, half_width_m: float = 12.0, bin_m: float = 0.05):
    """Histogram of path offsets: (bin centres, counts), for find_peaks."""
    edges = np.arange(-half_width_m, half_width_m + bin_m / 2, bin_m)
    counts, _ = np.histogram(offsets[np.abs(offsets) <= half_width_m], edges)
    return (edges[:-1] + edges[1:]) / 2, counts
