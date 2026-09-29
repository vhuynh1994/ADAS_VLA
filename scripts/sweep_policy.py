#!/usr/bin/env python3
"""Offline sweep of the cautious action policy using the action probabilities logged by `adas-vla eval`.

The longitudinal action is re-chosen from the logged probabilities; the lateral action is kept from the
greedy run (it is decoded after the longitudinal token and rarely changes). Thresholds are selected with a
2-fold split by route/clip (select on one half, measure on the other) so the reported numbers are not tuned
on the samples they are measured on.

  python scripts/sweep_policy.py outputs/eval_v3p_val.jsonl --data data/ds_v2/labels.jsonl [--nexar ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adas_vla.reasoning.vlm import LONG_ORDER, choose_action  # noqa: E402
from adas_vla.training.data import load_records  # noqa: E402

RANK = {a: i for i, a in enumerate(LONG_ORDER)}
GRID = [round(0.05 * i, 2) for i in range(2, 11)] + [1.01]  # 0.10 .. 0.50, and "off"


def metrics(rows: list[dict], tau_d: float, tau_b: float) -> dict:
    n = ok = under = over = 0
    for r in rows:
        if not r.get("probs"):
            continue
        g = r["gt"]["longitudinal"]
        p = choose_action(r["probs"], "cautious", tau_d, tau_b)
        n += 1
        ok += p == g and r["pred"]["lateral"] == r["gt"]["lateral"]
        under += RANK[g] >= RANK["DECELERATE"] and RANK[p] < RANK[g]
        over += RANK[p] > RANK[g]
    return {"n": n, "joint": ok / max(1, n), "under": under / max(1, n), "over": over / max(1, n)}


def select(rows: list[dict], max_under: float) -> tuple[float, float]:
    """Most accurate thresholds whose under-braking is <= max_under (else the lowest under-braking)."""
    scored = [((td, tb), metrics(rows, td, tb)) for td, tb in product(GRID, GRID)]
    feasible = [s for s in scored if s[1]["under"] <= max_under]
    if feasible:
        return max(feasible, key=lambda s: (s[1]["joint"], -s[1]["under"]))[0]
    return min(scored, key=lambda s: (s[1]["under"], -s[1]["joint"]))[0]


def fmt(m: dict) -> str:
    return f"joint {m['joint']:.1%} | under-braking {m['under']:.1%} | over-braking {m['over']:.1%} (n={m['n']})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--data", required=True)
    ap.add_argument("--nexar", help="eval report on test_nexar, to check the chosen thresholds")
    ap.add_argument("--max-under", type=float, default=0.01)
    args = ap.parse_args()

    group = {r["image"]: r.get("group", "?") for r in load_records(args.data, include_excluded=True)}
    rows = [json.loads(line) for line in open(args.report) if line.strip()]
    fold = lambda r: int(hashlib.md5(group.get(r["image"], "?").encode()).hexdigest(), 16) % 2
    folds = [[r for r in rows if fold(r) == k] for k in (0, 1)]

    print("greedy (policy off):", fmt(metrics(rows, 1.01, 1.01)))
    print("\nTrade-off (tau_decel = tau_brake = t):")
    for t in GRID[:-1]:
        print(f"  t={t:.2f}: {fmt(metrics(rows, t, t))}")

    print(f"\n2-fold selection (target under-braking <= {args.max_under:.0%}):")
    agg = {"n": 0, "ok": 0.0, "under": 0.0, "over": 0.0}
    for k in (0, 1):
        td, tb = select(folds[k], args.max_under)
        m = metrics(folds[1 - k], td, tb)
        print(f"  select on fold {k} -> tau_decel={td}, tau_brake={tb}; measured on fold {1 - k}: {fmt(m)}")
        agg["n"] += m["n"]
        for key, name in (("ok", "joint"), ("under", "under"), ("over", "over")):
            agg[key] += m[name] * m["n"]
    n = agg["n"]
    print(f"  => held-out estimate: joint {agg['ok'] / n:.1%} | under-braking {agg['under'] / n:.1%} "
          f"| over-braking {agg['over'] / n:.1%}")
    td, tb = select(rows, args.max_under)
    print(f"\nThresholds selected on all of val (for deployment): tau_decel={td}, tau_brake={tb}")
    if args.nexar:
        nx = [json.loads(line) for line in open(args.nexar) if line.strip()]
        print("Nexar crash test, greedy:  ", fmt(metrics(nx, 1.01, 1.01)))
        print("Nexar crash test, cautious:", fmt(metrics(nx, td, tb)))


if __name__ == "__main__":
    main()
