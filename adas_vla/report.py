"""Self-contained HTML report comparing evaluation runs (e.g. zero-shot baseline vs fine-tuned)."""

from __future__ import annotations

import base64
import html
import json
from collections import Counter
from pathlib import Path

import cv2

from .reasoning.prompts import VLM_LAT_ACTIONS, VLM_LONG_ACTIONS
from .training.evaluate import wilson_interval

ACCEPTANCE = [  # (metric, label, target, higher_is_better)
    ("joint_accuracy", "Joint action accuracy", 0.85, True),
    ("under_braking_rate", "Under-braking rate (VLM alone)", 0.01, False),
    ("system_under_braking_rate", "Under-braking rate (VLM + safety gate)", 0.01, False),
    ("json_valid_rate", "Valid JSON", 1.0, True),
    ("latency_p50_s", "Latency p50 (s)", 1.0, False),
]


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> dict:
    """Recompute metrics from a per-sample eval report (adas-vla eval --report)."""
    from .types import LongAction

    n = len(rows)
    valid = [r for r in rows if r["pred"]]
    replayed = [r for r in rows if r.get("final")]  # eval reports written with stored `lead` perception
    def ok(r, k):
        return r["pred"] and r["pred"][k] == r["gt"][k]

    def under(r, key):
        gt, pred = LongAction(r["gt"]["longitudinal"]), LongAction(r[key]["longitudinal"])
        return gt.rank >= LongAction.DECELERATE.rank and pred.rank < gt.rank

    counts = {
        "joint_accuracy": sum(bool(ok(r, "longitudinal") and ok(r, "lateral")) for r in rows),
        "longitudinal_accuracy": sum(bool(ok(r, "longitudinal")) for r in rows),
        "lateral_accuracy": sum(bool(ok(r, "lateral")) for r in rows),
        "under_braking_rate": sum(under(r, "pred") for r in valid),
    }
    lat = sorted(r["latency_s"] for r in rows)
    metrics = {"samples": n, "json_valid_rate": len(valid) / max(1, n)}
    for key, k in counts.items():
        metrics[key] = k / max(1, n)
        metrics[key + "_ci95"] = wilson_interval(k, n)
    system_under = sum(under(r, "final") for r in replayed)
    metrics["system_under_braking_rate"] = system_under / len(replayed) if replayed else None
    metrics["system_under_braking_rate_ci95"] = wilson_interval(system_under, len(replayed)) if replayed else None
    metrics["replayed_samples"] = len(replayed)
    metrics["latency_p50_s"] = lat[len(lat) // 2] if lat else 0.0
    metrics["latency_p90_s"] = lat[min(len(lat) - 1, int(len(lat) * 0.9))] if lat else 0.0
    return metrics


def _rate_cell(metrics: dict, key: str, css: str = "") -> str:
    """'72.6% (69.6–75.4)' with the 95% interval, or '—' when the metric is not available for this run."""
    v = metrics.get(key)
    if v is None:
        return "<td class='muted'>—</td>"
    ci = metrics.get(key + "_ci95")
    span = f" <span class='muted'>({ci[0]:.1%}–{ci[1]:.1%})</span>" if ci else ""
    return f"<td class='{css}'>{v:.1%}{span}</td>"


def _thumb(path: Path, width: int = 360) -> str:
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, round(h * width / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _confusion(rows: list[dict], key: str, labels: list[str]) -> str:
    c = Counter((r["gt"][key], r["pred"][key] if r["pred"] else "INVALID") for r in rows)
    cols = labels + (["INVALID"] if any(p == "INVALID" for _, p in c) else [])
    head = "".join(f"<th>{html.escape(x)}</th>" for x in cols)
    body = ""
    for g in labels:
        total = sum(c[(g, p)] for p in cols)
        if not total:
            continue
        cells = "".join(
            f"<td class='{'diag' if g == p else ('err' if c[(g, p)] else '')}'>{c[(g, p)] or ''}</td>" for p in cols)
        body += f"<tr><th>{html.escape(g)}</th>{cells}<td class='muted'>{total}</td></tr>"
    return f"<table class='cm'><tr><th>truth \\ pred</th>{head}<th>n</th></tr>{body}</table>"


def build_report(runs: dict[str, str], data_dir: str, out: str, title: str, notes: str = "",
                 examples: int = 12) -> None:
    loaded = {name: _load(path) for name, path in runs.items()}
    metrics = {name: summarize(rows) for name, rows in loaded.items()}
    names = list(runs)

    rows_html = ""
    for key, label, target, higher in ACCEPTANCE:
        if all(metrics[name].get(key) is None for name in names):
            continue  # e.g. no run was evaluated on a dataset with stored perception
        cells = ""
        for name in names:
            v = metrics[name].get(key)
            if v is None:
                cells += "<td class='muted'>—</td>"
                continue
            passed = v >= target if higher else v <= target
            css = "pass" if passed else "fail"
            mark = "✓" if passed else "✗"
            if key.startswith("latency"):
                cells += f"<td class='{css}'>{v:.2f} s {mark}</td>"
            else:
                cells += _rate_cell(metrics[name], key, css).replace("</td>", f" {mark}</td>")
        tshown = f"≤ {target:.1f} s" if key.startswith("latency") else (
            f"{target:.0%}" if target == 1.0 else f"≥ {target:.0%}" if higher else f"≤ {target:.0%}")
        rows_html += f"<tr><th>{label}</th><td class='muted'>{tshown}</td>{cells}</tr>"
    for key, label in (("longitudinal_accuracy", "Longitudinal accuracy"), ("lateral_accuracy", "Lateral accuracy"),
                       ("latency_p90_s", "Latency p90 (s)"), ("samples", "Samples"),
                       ("replayed_samples", "Samples with stored perception (gate replay)")):
        cells = "".join(
            f"<td>{metrics[n][key]:.2f} s</td>" if key.startswith("latency") else
            (f"<td>{metrics[n][key]}</td>" if key.endswith("samples") else _rate_cell(metrics[n], key))
            for n in names)
        rows_html += f"<tr><th>{label}</th><td></td>{cells}</tr>"
    # Per-source breakdown (joint accuracy / under-braking), from the dataset's labels.jsonl.
    source_of = {}
    labels_path = Path(data_dir) / "labels.jsonl"
    if labels_path.exists():
        with labels_path.open() as f:
            source_of = {r["image"]: r.get("source", "?") for r in map(json.loads, f) if r.get("image")}
    sources = sorted({source_of.get(r["image"], "?") for rows in loaded.values() for r in rows})
    if len(sources) > 1:
        for src in sources:
            cells = ""
            for n in names:
                sub_m = summarize([r for r in loaded[n] if source_of.get(r["image"], "?") == src])
                cells += f"<td>{sub_m['joint_accuracy']:.1%} / {sub_m['under_braking_rate']:.1%}</td>"
            count = sum(1 for r in loaded[names[0]] if source_of.get(r["image"], "?") == src)
            rows_html += f"<tr><th>{html.escape(src)} (n={count})</th><td class='muted'>acc / under-brake</td>{cells}</tr>"
    head_cells = "".join(f"<th>{html.escape(n)}</th>" for n in names)

    cms = ""
    for name in names:
        cms += (f"<section class='card'><h3>{html.escape(name)}</h3><h4>Longitudinal</h4>"
                f"{_confusion(loaded[name], 'longitudinal', VLM_LONG_ACTIONS)}<h4>Lateral</h4>"
                f"{_confusion(loaded[name], 'lateral', VLM_LAT_ACTIONS)}</section>")

    # Examples: where the last run fixed the first run, then remaining errors of the last run.
    first, last = names[0], names[-1]
    by_img = {name: {r["image"]: r for r in loaded[name]} for name in names}

    def correct(r):
        return bool(r and r["pred"] and r["pred"]["longitudinal"] == r["gt"]["longitudinal"]
                    and r["pred"]["lateral"] == r["gt"]["lateral"])

    fixed = [img for img in by_img[last] if correct(by_img[last][img]) and not correct(by_img[first].get(img))]
    wrong = [img for img in by_img[last] if not correct(by_img[last][img])]
    cards = ""
    for title_, imgs in ((f"Fixed by {last}", fixed[:examples]), (f"Still wrong in {last}", wrong[:examples])):
        if not imgs:
            continue
        cards += f"<h2>{html.escape(title_)}</h2><div class='grid'>"
        for img in imgs:
            gt = by_img[last][img]["gt"]
            lines = f"<div><b>Truth</b> {gt['longitudinal']} / {gt['lateral']}</div>"
            for name in names:
                r = by_img[name].get(img)
                p = r["pred"] if r else None
                txt = f"{p['longitudinal']} / {p['lateral']}" if p else "invalid"
                cls = "ok" if correct(r) else "bad"
                reason = html.escape(p["reason"]) if p and p.get("reason") else ""
                lines += f"<div class='{cls}'><b>{html.escape(name)}</b> {txt}<br><span class='muted'>{reason}</span></div>"
            cards += (f"<figure><img src='{_thumb(Path(data_dir) / img)}' alt=''>"
                      f"<figcaption>{lines}</figcaption></figure>")
        cards += "</div>"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --text:#1d2330; --muted:#667085; --line:#d9dde5; --ok:#16794a; --bad:#c0362c;
        --okbg:#e7f5ee; --badbg:#fdecea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#12151b; --panel:#1b2029; --text:#e6e9ef; --muted:#98a2b3;
        --line:#2c3340; --ok:#4cc38a; --bad:#ff7b72; --okbg:#153325; --badbg:#3a1d1b; }} }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1200px; margin:auto; padding:24px 16px 48px; }} h1 {{ margin:0 0 4px; font-size:22px; }}
h2 {{ font-size:17px; margin:28px 0 10px; }} h3 {{ margin:0 0 6px; }} h4 {{ margin:10px 0 4px; color:var(--muted); font-size:12px; text-transform:uppercase; }}
.muted {{ color:var(--muted); }} .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; overflow-x:auto; }}
table {{ border-collapse:collapse; }} th, td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; }}
td.pass {{ color:var(--ok); font-weight:600; }} td.fail {{ color:var(--bad); font-weight:600; }}
.cm td, .cm th {{ text-align:center; font-size:12.5px; padding:4px 7px; }} .cm td.diag {{ background:var(--okbg); }}
.cm td.err {{ background:var(--badbg); }} .cms {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:12px; }}
figure {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
figure img {{ width:100%; display:block; }} figcaption {{ padding:8px 10px; font-size:12.5px; }}
.ok b {{ color:var(--ok); }} .bad b {{ color:var(--bad); }} .notes {{ white-space:pre-wrap; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="muted">Evaluated on the held-out val split (whole videos never seen in training).
Rates show their 95% Wilson interval; two runs whose intervals overlap widely are not distinguishable on this val set.</p>
<div class="card"><table><tr><th>Metric</th><th>Target</th>{head_cells}</tr>{rows_html}</table></div>
{f"<h2>Notes</h2><div class='card notes'>{html.escape(notes)}</div>" if notes else ""}
<h2>Confusion matrices</h2><div class="cms">{cms}</div>
{cards}
</main></body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(page)
    print(f"Report: {out}")
    for name in names:
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics[name].items()}))
