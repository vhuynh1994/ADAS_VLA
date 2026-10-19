#!/usr/bin/env python3
"""Repeatability of two route lane maps of the same road (scripts/build_lane_map.py, second one built with
--frame-from the first): along the stretches both drives covered, the lane-marking lines of drive B are located
across drive A's path and matched to drive A's lines. The offsets between matched lines measure how well two
independent drives (GNSS/INS pose + monocular projection) agree, with no external map.

  python scripts/compare_lane_maps.py outputs/lane_map/2018-07-27--06-03-57 outputs/lane_map/2018-07-30--13-03-07

Writes <A>/compare_<B>.json and compare_<B>.png (one stretch with both drives straightened along A's path, and the
line offsets along the route).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adas_vla.hud import _font  # noqa: E402
from adas_vla.mapping import find_peaks, offset_profile, path_offsets  # noqa: E402


def stretch(points, traj, heads, dist, lo, hi):
    """Points of `points` within A's stretch [lo, hi): (along m from lo, offset m across A's path)."""
    centre = traj[(lo + hi) // 2]
    near = np.linalg.norm(points - centre, axis=1) < (dist[hi] - dist[lo]) / 2 + 30
    pidx, off, along = path_offsets(points[near], traj[lo:hi], heads[lo:hi])
    s = dist[lo:hi][pidx] + along - dist[lo]
    inside = (s >= 0) & (s <= dist[hi] - dist[lo]) & (np.abs(off) <= 12)
    return s[inside], off[inside]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--step", type=float, default=250.0, help="metres between compared stretches")
    ap.add_argument("--length", type=float, default=120.0, help="stretch length (m)")
    ap.add_argument("--match", type=float, default=1.0, help="max offset (m) to pair a line of B with one of A")
    args = ap.parse_args()
    A, B = np.load(Path(args.a) / "points.npz"), np.load(Path(args.b) / "points.npz")
    if not np.allclose(A["ref_ecef"], B["ref_ecef"]):
        raise SystemExit("the maps use different local frames: build B with --frame-from A/points.npz")
    pa, pb = A["points"].astype(float), B["points"].astype(float)
    ta, ha = A["traj"], A["heads"]
    tb = B["traj"]
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ta, axis=0), axis=1))])
    rows, example = [], None
    for c in np.arange(args.length, dist[-1] - args.length, args.step):
        lo, hi = np.searchsorted(dist, c - args.length / 2), np.searchsorted(dist, c + args.length / 2)
        mid = ta[(lo + hi) // 2]
        if np.min(np.linalg.norm(tb[::5] - mid, axis=1)) > 15:  # B did not drive here
            continue
        sa, oa = stretch(pa, ta, ha, dist, lo, hi)
        sb, ob = stretch(pb, ta, ha, dist, lo, hi)
        if len(oa) < 2000 or len(ob) < 2000:
            continue
        peaks_a = find_peaks(*offset_profile(oa))
        peaks_b = find_peaks(*offset_profile(ob))
        pairs = []
        for o_a, w_a, _ in peaks_a:
            best = min(peaks_b, key=lambda p: abs(p[0] - o_a), default=None)
            if best is not None and abs(best[0] - o_a) <= args.match:
                pairs.append({"a_m": round(o_a, 2), "b_m": round(best[0], 2), "delta_m": round(best[0] - o_a, 3),
                              "fwhm_a_m": round(w_a, 2), "fwhm_b_m": round(best[1], 2)})
        rows.append({"km": round(float(c) / 1000, 2), "pairs": pairs, "lines_a": len(peaks_a), "lines_b": len(peaks_b)})
        if example is None and len(pairs) >= 2:
            example = (float(c), sa, oa, sb, ob, pairs)
    deltas = np.array([p["delta_m"] for r in rows for p in r["pairs"]])
    summary = {"a": Path(args.a).name, "b": Path(args.b).name, "stretches": len(rows),
               "matched_lines": int(len(deltas)),
               "median_abs_delta_m": round(float(np.median(np.abs(deltas))), 3) if len(deltas) else None,
               "p90_abs_delta_m": round(float(np.percentile(np.abs(deltas), 90)), 3) if len(deltas) else None,
               "mean_delta_m": round(float(np.mean(deltas)), 3) if len(deltas) else None,
               "median_fwhm_m": round(float(np.median([p["fwhm_a_m"] for r in rows for p in r["pairs"]])), 2)
               if len(deltas) else None}
    out = Path(args.a)
    (out / f"compare_{Path(args.b).name}.json").write_text(json.dumps({"summary": summary, "stretches": rows}, indent=1))
    print(json.dumps(summary, indent=2))
    if example is not None:
        draw_example(example, rows, out / f"compare_{Path(args.b).name}.png", args.length, Path(args.a).name,
                     Path(args.b).name)


def draw_example(example, rows, path: Path, length: float, name_a: str, name_b: str) -> None:
    c, sa, oa, sb, ob, pairs = example
    m = 0.1
    w, h = int(length / m), int(24 / m)
    img = np.full((h, w, 3), 255, np.uint8)
    for s, o, color in ((sa, oa, (40, 40, 40)), (sb, ob, (0, 120, 255))):
        cnt = np.zeros((h, w), np.float32)
        np.add.at(cnt, (np.clip(((o + 12) / m).astype(int), 0, h - 1), np.clip((s / m).astype(int), 0, w - 1)), 1)
        mask = cnt > np.percentile(cnt[cnt > 0], 30) if (cnt > 0).any() else cnt > 0
        img[mask] = color
    cv2.line(img, (0, h // 2), (w, h // 2), (200, 140, 0), 1)
    # offsets along the route: +-1.5 m, grid every 0.5 m, a header band for the caption
    head, ph, pw, k = 28, 300, w, 100.0  # k = px per metre
    plot = np.full((head + ph, pw, 3), 255, np.uint8)
    zero = head + ph // 2
    for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = int(zero - v * k)
        cv2.line(plot, (50, y), (pw - 10, y), (150, 150, 150) if v == 0 else (225, 225, 225), 1)
        cv2.putText(plot, f"{v:+.1f}", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    kms = [r["km"] for r in rows for _ in r["pairs"]]
    ds = [p["delta_m"] for r in rows for p in r["pairs"]]
    if kms:
        k0, k1 = min(kms), max(kms)
        for km, d in zip(kms, ds):
            x = int(50 + (km - k0) / max(k1 - k0, 1e-6) * (pw - 70))
            y = int(np.clip(zero - d * k, head + 4, head + ph - 4))
            cv2.circle(plot, (x, y), 3, (0, 120, 255), -1, cv2.LINE_AA)
    canvas = np.vstack([img, np.full((8, w, 3), 255, np.uint8), plot])
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    f14, f16 = _font(14), _font(16)
    dr.text((8, 6), f"km {c / 1000:.2f}: vạch làn của 2 lần chạy, đường duỗi theo quỹ đạo lần A ({length:.0f} m)",
            font=f16, fill=(0, 0, 0))
    dr.text((8, 28), f"đen = {name_a}   cam = {name_b}   lệch: " + ", ".join(f"{p['delta_m'] * 100:+.0f} cm" for p in pairs),
            font=f14, fill=(0, 0, 0))
    dr.text((8, h + 8 + 6), f"độ lệch vạch làn B so với A dọc tuyến, km {min(kms) if kms else 0:.1f} → "
            f"{max(kms) if kms else 0:.1f} (m, lưới 0.5 m)", font=f14, fill=(0, 0, 0))
    pil.save(path)


if __name__ == "__main__":
    main()
