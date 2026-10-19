#!/usr/bin/env python3
"""Route lane map from a comma2k19 drive (map step 2): YOLOP lane-marking pixels of every frame projected to the
road (flat-road model of bev.py, horizon = lane vanishing point), placed with the drive's global pose and
accumulated over the route.

  python scripts/build_lane_map.py --list                                   # routes of a chunk: length, start/end
  python scripts/build_lane_map.py --route 2018-07-27--06-03-57 --out outputs/lane_map

Writes <out>/<route>/: overview.png (whole route), zoom_1..3.png (straight stretches, road left to right, with the
across-road profile of the markings), points.npz, route.geojson (trajectory + thinned marking points as lon/lat,
for QGIS / geojson.io over satellite imagery) and report.json (sharpness and spacing of the marking lines).
Camera: comma EON road camera, 1164x874, focal ~910 px (hfov 65 deg); height above the road --cam-height.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adas_vla.config import CameraConfig, load_config  # noqa: E402
from adas_vla.datasets.comma2k19 import _load_npy, list_segments  # noqa: E402
from adas_vla.hud import _font  # noqa: E402
from adas_vla.mapping import (LocalFrame, camera_heading, find_peaks, heading_of, offset_profile,  # noqa: E402
                              path_offsets, wrap)
from adas_vla.perception.geometry import MotionEstimator, focal_length_px  # noqa: E402
from adas_vla.perception.lanes import build_lane_detector  # noqa: E402

ZIP = ROOT / "data/raw/commaai--comma2k19/raw_data/Chunk_1.zip"


def routes_of(zip_path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for seg in list_segments(zip_path):
        route = seg.rstrip("/").rsplit("/", 1)[0]
        out.setdefault(route, []).append(seg)
    for r in out:
        out[r].sort(key=lambda s: int(s.rstrip("/").rsplit("/", 1)[1]))
    return out


def list_routes(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for route, segs in routes_of(zip_path).items():
            pos = [_load_npy(z, s + "global_pose/frame_positions") for s in (segs[0], segs[-1])]
            frame = LocalFrame(pos[0][0])
            a, b = frame.to_lla(np.zeros((1, 2)))[0], frame.to_lla(frame.to_enu(pos[1][-1])[:, :2])[0]
            print(f"{route.split('/', 1)[1]:42s} {len(segs):2d} min  start {a[0]:.4f},{a[1]:.4f}  end {b[0]:.4f},{b[1]:.4f}")


def choose_convention(q: np.ndarray, v_en: np.ndarray, frame: LocalFrame) -> tuple[bool, float]:
    """Quaternion order whose camera heading agrees with the direction of travel; returns (scalar_first, median
    camera-minus-travel heading in degrees, i.e. the mounting yaw offset)."""
    moving = np.linalg.norm(v_en[:, :2], axis=1) > 5.0
    travel = heading_of(v_en[moving])
    best = None
    for scalar_first in (True, False):
        diff = wrap(camera_heading(q[moving], frame, scalar_first) - travel)
        med = float(np.median(np.abs(diff)))
        if best is None or med < best[2]:
            best = (scalar_first, float(np.degrees(np.median(diff))), med)
    return best[0], best[1]


def density_image(points: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    h, _, _ = np.histogram2d(points[:, 0], points[:, 1], [x_edges, y_edges])
    if h.max() == 0:
        return np.zeros(h.shape, np.uint8)
    img = np.log1p(h) / np.log1p(np.percentile(h[h > 0], 99))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_overview(points, traj, windows, out: Path) -> None:
    lo, hi = traj.min(0) - 60, traj.max(0) + 60
    m_per_px = max((hi - lo).max() / 1800.0, 0.5)
    xe = np.arange(lo[0], hi[0] + m_per_px, m_per_px)
    ye = np.arange(lo[1], hi[1] + m_per_px, m_per_px)
    dens = density_image(points, xe, ye).T[::-1]  # rows = north (top) to south
    img = cv2.cvtColor(255 - dens, cv2.COLOR_GRAY2BGR)

    def px(p):
        return int((p[0] - lo[0]) / m_per_px), int((hi[1] - p[1]) / m_per_px)

    cv2.polylines(img, [np.array([px(p) for p in traj[::5]], np.int32)], False, (200, 140, 0), 2, cv2.LINE_AA)
    cv2.circle(img, px(traj[0]), 9, (60, 170, 60), -1)
    cv2.circle(img, px(traj[-1]), 9, (40, 40, 210), -1)
    for k, i in enumerate(windows, 1):
        cv2.circle(img, px(traj[i]), 18, (0, 120, 255), 3)
        cv2.putText(img, str(k), (px(traj[i])[0] + 22, px(traj[i])[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 120, 255), 2)
    bar_m = 10 ** math.floor(math.log10(300 * m_per_px))
    x0, y0 = 30, img.shape[0] - 40
    cv2.line(img, (x0, y0), (x0 + int(bar_m / m_per_px), y0), (0, 0, 0), 4)
    cv2.putText(img, f"{bar_m:.0f} m", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.arrowedLine(img, (img.shape[1] - 50, 110), (img.shape[1] - 50, 40), (0, 0, 0), 3, tipLength=0.3)
    cv2.putText(img, "N", (img.shape[1] - 60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(str(out), img)


def stable_windows(dist: np.ndarray, offset_norm: np.ndarray, n: int = 3, length_m: float = 200.0,
                   min_gap_m: float = 1000.0) -> list[tuple[int, int]]:
    """Up to n stretches [lo, hi) of the route, `length_m` long, where the car kept its lane (ego-lane offset seen in
    >= 80 % of the sampled frames, range < 0.5 half-lane widths), the steadiest first, `min_gap_m` apart."""
    cands = []
    for lo in range(0, len(dist), 40):
        hi = int(np.searchsorted(dist, dist[lo] + length_m))
        if hi >= len(dist):
            break
        o = offset_norm[lo:hi]
        seen = o[np.isfinite(o)]  # sampled frames with an ego lane (inf = not sampled, nan = no ego lane)
        sampled = np.count_nonzero(~np.isinf(o))
        if len(seen) == 0 or sampled == 0:
            continue
        if len(seen) >= 0.8 * sampled and np.ptp(seen) < 0.5:
            cands.append((float(np.std(seen)), lo, hi))
    picked: list[tuple[int, int]] = []
    for _, lo, hi in sorted(cands):
        if all(abs(dist[lo] - dist[a]) >= min_gap_m for a, _ in picked):
            picked.append((lo, hi))
        if len(picked) == n:
            break
    return sorted(picked)


def render_strip(s_m, off_m, peaks, centres, counts, out: Path, title: str, length_m: float, half_w=12.0,
                 m_per_px=0.1) -> None:
    """The stretch with the road straightened: x = distance along the car's path, y = offset across it."""
    xe = np.arange(0, length_m + m_per_px, m_per_px)
    ye = np.arange(-half_w, half_w + m_per_px, m_per_px)
    dens = density_image(np.stack([s_m, off_m], 1), xe, ye).T  # rows: left of travel (top) to right
    road = cv2.cvtColor(255 - dens, cv2.COLOR_GRAY2BGR)
    mid = road.shape[0] // 2
    cv2.line(road, (0, mid), (road.shape[1], mid), (200, 140, 0), 1)  # the car's path
    for x_m in range(0, int(length_m) + 1, 10):
        cv2.line(road, (int(x_m / m_per_px), 0), (int(x_m / m_per_px), 8), (120, 120, 120), 1)
    prof_h, prof_w = road.shape[0], 360
    prof = np.full((prof_h, prof_w, 3), 255, np.uint8)
    y = (centres + half_w) / (2 * half_w) * prof_h
    xv = counts / max(counts.max(), 1) * (prof_w - 20)
    cv2.polylines(prof, [np.stack([xv, y], 1).astype(np.int32)], False, (60, 60, 60), 1, cv2.LINE_AA)
    canvas = np.hstack([road, np.full((prof_h, 6, 3), 255, np.uint8), prof])
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    f14, f16 = _font(14), _font(16)
    draw.text((8, 8), title, font=f16, fill=(0, 0, 0))
    for off, fwhm, _ in peaks:
        yy = (off + half_w) / (2 * half_w) * prof_h
        draw.line([(road.shape[1] + 6, yy), (canvas.shape[1], yy)], fill=(230, 120, 0), width=1)
        draw.text((road.shape[1] + 110, yy - 16), f"{off:+.2f} m · nét {fwhm * 100:.0f} cm", font=f14, fill=(200, 90, 0))
    gaps = [b[0] - a[0] for a, b in zip(peaks, peaks[1:])]
    if gaps:
        draw.text((8, prof_h - 24), "khoảng cách giữa các vạch: " + ", ".join(f"{g:.2f} m" for g in gaps),
                  font=f14, fill=(0, 0, 0))
    pil.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=str(ZIP))
    ap.add_argument("--list", action="store_true", help="list the routes of the chunk and exit")
    ap.add_argument("--route", help="route name or a unique part of it (e.g. 2018-07-27--06-03-57)")
    ap.add_argument("--max-segments", type=int, default=0, help="first N one-minute segments only (0 = all)")
    ap.add_argument("--every", type=int, default=2, help="use every N-th frame (20 fps video)")
    ap.add_argument("--cam-height", type=float, default=1.22, help="camera height above the road (m)")
    ap.add_argument("--lane-width", type=float, default=None,
                    help="calibrate the camera height so the median ego lane is this wide 9 m ahead (US: 3.66)")
    ap.add_argument("--hfov", type=float, default=65.0, help="horizontal field of view (deg); comma EON ~65")
    ap.add_argument("--z-range", type=float, nargs=2, default=(4.0, 15.0), help="road points used, m ahead")
    ap.add_argument("--x-max", type=float, default=10.0, help="road points used, m to each side")
    ap.add_argument("--max-points", type=int, default=800, help="marking pixels kept per frame")
    ap.add_argument("--frame-from", help="points.npz of another map: use its local frame (to compare two drives)")
    ap.add_argument("--out", default=str(ROOT / "outputs/lane_map"))
    args = ap.parse_args()
    zip_path = Path(args.zip)
    if args.list:
        list_routes(zip_path)
        return
    routes = routes_of(zip_path)
    match = [r for r in routes if args.route and args.route in r]
    if len(match) != 1:
        raise SystemExit(f"--route must match exactly one route, got {len(match)}; see --list")
    route = match[0]
    segs = routes[route][: args.max_segments or None]
    out = Path(args.out) / route.split("|")[-1]
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    det = build_lane_detector(cfg.perception.lane_model, cfg.perception.lane_weights, cfg.perception.device)
    motion = MotionEstimator(CameraConfig(hfov_deg=args.hfov, mount_height_m=args.cam_height))
    rng = np.random.default_rng(0)
    h0 = args.cam_height  # nominal height: only selects which rows are kept (z-range); points stored per metre of height
    unit_pts, traj, heads, offset_norm = [], [], [], []
    widths1 = {6: [], 9: [], 12: []}  # ego-lane width per metre of camera height, at nominal distances
    frame = conv = None
    base = 0
    t0 = time.perf_counter()
    with zipfile.ZipFile(zip_path) as z:
        for si, seg in enumerate(segs):
            pos = _load_npy(z, seg + "global_pose/frame_positions")
            quat = _load_npy(z, seg + "global_pose/frame_orientations")
            vel = _load_npy(z, seg + "global_pose/frame_velocities")
            times = _load_npy(z, seg + "global_pose/frame_times")
            if frame is None:
                frame = LocalFrame(np.load(args.frame_from)["ref_ecef"] if args.frame_from else pos[0])
            p_en = frame.to_enu(pos)[:, :2]
            if conv is None:
                conv = choose_convention(quat, frame.vec_to_enu(vel), frame)
                print(f"quaternion scalar-first: {conv[0]}; camera yaw vs travel: {conv[1]:+.2f} deg")
            head = camera_heading(quat, frame, conv[0])
            onorm = np.full(len(pos), np.inf)  # inf = frame not sampled, nan = sampled without an ego lane
            video = out / "segment.hevc"
            video.write_bytes(z.read(seg + "video.hevc"))
            cap = cv2.VideoCapture(str(video))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # a raw .hevc stream has no container: count is junk
            n = min(len(pos), count) if count > 0 else len(pos)
            for i in range(n):
                ok, img = cap.read()
                if not ok:
                    break
                if i % args.every:
                    continue
                h, w = img.shape[:2]
                lanes = det(img)
                hz = motion.horizon_row(lanes, float(times[i]), h)
                f_px = focal_length_px(w, args.hfov)
                onorm[i] = lanes.offset_norm if lanes.offset_norm is not None else np.nan
                ys, xs = np.nonzero(det.last_mask)
                keep = ys > hz + 0.03 * h
                ys, xs = ys[keep], xs[keep]
                if len(ys) > args.max_points:
                    sel = rng.choice(len(ys), args.max_points, replace=False)
                    ys, xs = ys[sel], xs[sel]
                dy = (ys - hz).astype(np.float64)
                z1, x1 = f_px / dy, (xs - w / 2) / dy  # metres per metre of camera height
                ok_pts = (z1 * h0 >= args.z_range[0]) & (z1 * h0 <= args.z_range[1]) & (np.abs(x1 * h0) <= args.x_max)
                if ok_pts.any():
                    unit_pts.append(np.stack([np.full(ok_pts.sum(), base + i), x1[ok_pts], z1[ok_pts]], 1))
                if lanes.valid:
                    for zn in widths1:
                        row = hz + f_px * h0 / zn
                        left, right = lanes.bounds_at(row)
                        widths1[zn].append((right - left) / (row - hz))
            cap.release()
            video.unlink()
            traj.append(p_en[:n])
            heads.append(head[:n])
            offset_norm.append(onorm[:n])
            base += n
            print(f"segment {si + 1}/{len(segs)} done, {sum(len(p) for p in unit_pts):,} points, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)

    traj, heads, offset_norm = np.concatenate(traj), np.concatenate(heads), np.concatenate(offset_norm)
    u = np.concatenate(unit_pts)
    idx = u[:, 0].astype(np.int64)
    med1 = {zn: float(np.median(v)) for zn, v in widths1.items() if v}
    cam_h = args.lane_width / med1[9] if args.lane_width and 9 in med1 else args.cam_height
    s_h, c_h = np.sin(heads[idx]), np.cos(heads[idx])
    xr, za = u[:, 1] * cam_h, u[:, 2] * cam_h
    points = np.stack([traj[idx, 0] + za * s_h + xr * c_h, traj[idx, 1] + za * c_h - xr * s_h], 1)
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))])
    np.savez_compressed(out / "points.npz", points=points.astype(np.float32), traj=traj, heads=heads,
                        ref_ecef=frame.ref, cam_height=cam_h)
    windows = stable_windows(dist, offset_norm)
    render_overview(points, traj, [lo + (hi - lo) // 2 for lo, hi in windows], out / "overview.png")
    report = {"route": route, "segments": len(segs), "length_km": round(float(dist[-1]) / 1000, 2),
              "points": int(len(points)), "frames_every": args.every, "hfov_deg": args.hfov,
              "cam_height_m": round(cam_h, 3), "cam_height_from": "lane width" if args.lane_width else "argument",
              "quaternion_scalar_first": conv[0], "camera_yaw_vs_travel_deg": round(conv[1], 2),
              "ego_lane_width_m": {f"{zn}m": round(v * cam_h, 2) for zn, v in med1.items()},
              "windows": []}
    for k, (lo, hi) in enumerate(windows, 1):
        sel = (idx >= lo) & (idx < hi)
        pidx, off, along = path_offsets(points[sel], traj[lo:hi], heads[lo:hi])
        s_m = dist[lo:hi][pidx] + along - dist[lo]
        inside = (s_m >= 0) & (s_m <= dist[hi] - dist[lo])  # points beyond the stretch ends snap to its end samples
        s_m, off = s_m[inside], off[inside]
        centres, counts = offset_profile(off)
        peaks = find_peaks(centres, counts)
        lat, lon, _ = frame.to_lla(traj[(lo + hi) // 2][None])[0]
        render_strip(s_m, off, peaks, centres, counts, out / f"zoom_{k}.png",
                     f"đoạn {k} · {lat:.5f}, {lon:.5f} · {dist[hi] - dist[lo]:.0f} m, đường được duỗi thẳng, "
                     f"xe đi từ trái sang phải", float(dist[hi] - dist[lo]))
        report["windows"].append({"lat": round(lat, 6), "lon": round(lon, 6),
                                  "peaks": [{"offset_m": round(o, 2), "fwhm_m": round(f, 2)} for o, f, _ in peaks],
                                  "spacing_m": [round(b[0] - a[0], 2) for a, b in zip(peaks, peaks[1:])]})
    thin = points[rng.choice(len(points), min(len(points), 30000), replace=False)]
    lla_traj, lla_pts = frame.to_lla(traj[::10]), frame.to_lla(thin)
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "trajectory"},
         "geometry": {"type": "LineString", "coordinates": [[round(b, 7), round(a, 7)] for a, b, _ in lla_traj]}},
        {"type": "Feature", "properties": {"name": "lane markings (thinned)"},
         "geometry": {"type": "MultiPoint", "coordinates": [[round(b, 7), round(a, 7)] for a, b, _ in lla_pts]}}]}
    (out / "route.geojson").write_text(json.dumps(geo))
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
