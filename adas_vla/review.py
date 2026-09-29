"""Local web app for reviewing auto-generated labels quickly (keyboard driven).

Reviews are appended to <labels>.reviews.jsonl (last entry per id wins) and overlaid by load_records(),
so the auto labels stay untouched and every review is traceable.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .reasoning.prompts import VLM_LAT_ACTIONS, VLM_LONG_ACTIONS
from .training.data import load_records, reviews_path

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADAS Label Review</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --text:#1d2330; --muted:#667085; --line:#d9dde5; --accent:#2f6fed;
        --ok:#16794a; --warn:#b54708; --bad:#c0362c; --chip:#eef2f8; }
@media (prefers-color-scheme: dark) { :root { --bg:#12151b; --panel:#1b2029; --text:#e6e9ef; --muted:#98a2b3;
        --line:#2c3340; --accent:#6d9cff; --ok:#4cc38a; --warn:#f0a35b; --bad:#ff7b72; --chip:#252c38; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { display:flex; gap:16px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line);
  background:var(--panel); position:sticky; top:0; z-index:2; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; } .muted { color:var(--muted); }
select, input, button, textarea { font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line);
  border-radius:6px; padding:5px 8px; }
button { cursor:pointer; } button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
.progress { flex:1; min-width:160px; height:8px; background:var(--chip); border-radius:4px; overflow:hidden; }
.progress > div { height:100%; background:var(--ok); width:0; }
main { display:grid; grid-template-columns:minmax(0,1fr) 380px; gap:16px; padding:16px; max-width:1500px; margin:auto; }
@media (max-width: 900px) { main { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; }
img#frame { width:100%; border-radius:6px; display:block; background:#000; }
pre { white-space:pre-wrap; margin:0; font:12.5px/1.4 ui-monospace,Menlo,Consolas,monospace; }
.group { margin:10px 0; } .group h3 { margin:0 0 6px; font-size:12px; text-transform:uppercase;
  letter-spacing:.04em; color:var(--muted); }
.opts { display:flex; flex-wrap:wrap; gap:6px; }
.opt { border:1px solid var(--line); background:var(--chip); border-radius:6px; padding:5px 8px; cursor:pointer;
  font-size:13px; } .opt kbd { color:var(--muted); margin-right:4px; }
.opt.sel { background:var(--accent); color:#fff; border-color:var(--accent); } .opt.sel kbd { color:#dbe6ff; }
.meta { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
.chip { background:var(--chip); border-radius:999px; padding:2px 10px; font-size:12px; }
.chip.ok { color:var(--ok); } .chip.warn { color:var(--warn); }
.row { display:flex; gap:8px; align-items:center; } .row input[type=number] { width:90px; }
textarea { width:100%; min-height:54px; resize:vertical; }
.keys { font-size:12px; color:var(--muted); margin-top:8px; }
.toast { position:fixed; bottom:16px; right:16px; background:var(--ok); color:#fff; padding:8px 14px;
  border-radius:8px; opacity:0; transition:opacity .2s; } .toast.show { opacity:1; }
</style></head><body>
<header>
  <h1>ADAS Label Review</h1>
  <label>Show <select id="filter">
    <option value="unreviewed">Unreviewed</option><option value="val">Val split</option>
    <option value="train">Train split</option><option value="all">All</option></select></label>
  <label>Source <select id="source"><option value="">All sources</option></select></label>
  <div class="progress"><div id="bar"></div></div>
  <span id="count" class="muted"></span>
</header>
<main>
  <section class="card">
    <div class="meta" id="meta"></div>
    <img id="frame" alt="camera frame">
  </section>
  <section class="card">
    <div class="group"><h3>Longitudinal</h3><div class="opts" id="long"></div></div>
    <div class="group"><h3>Lateral</h3><div class="opts" id="lat"></div></div>
    <div class="group"><h3>Target speed and risk</h3>
      <div class="row"><input type="number" id="speed" min="0" max="200"> km/h
        <select id="risk"><option>low</option><option>medium</option><option>high</option></select></div></div>
    <div class="group"><h3>Reason (max 12 words)</h3><textarea id="reason"></textarea></div>
    <div class="row"><button id="prev">&larr; Prev</button><button class="primary" id="save">Save &amp; next (Enter)</button>
      <button id="next">Next &rarr;</button><button id="exclude">Exclude (X)</button></div>
    <div class="keys">Keys: 1-5 longitudinal · Q W E A D lateral · Enter save · X exclude (unusable) · ←/→ navigate · R edit reason · Esc leave text</div>
    <div class="group"><h3>Perception context</h3><pre id="context"></pre></div>
  </section>
</main>
<div class="toast" id="toast">Saved</div>
<script>
const LONG = __LONG__, LAT = __LAT__;
const LONG_KEYS = ["1","2","3","4","5"], LAT_KEYS = ["q","w","e","a","d"];
let items = [], view = [], pos = 0, cur = null;
const $ = id => document.getElementById(id);
function opts(el, values, keys, field) {
  el.innerHTML = "";
  values.forEach((v, i) => { const b = document.createElement("span"); b.className = "opt"; b.dataset.v = v;
    b.innerHTML = `<kbd>${keys[i].toUpperCase()}</kbd>${v}`; b.onclick = () => pick(field, v); el.appendChild(b); });
}
function pick(field, v) { if (!cur) return; cur.target[field] = v; paint(); }
function paint() {
  for (const [id, f] of [["long","longitudinal"],["lat","lateral"]])
    for (const b of $(id).children) b.classList.toggle("sel", b.dataset.v === cur.target[f]);
}
function applyFilter() {
  const f = $("filter").value, s = $("source").value;
  view = items.filter(it => (!s || it.source === s) &&
    (f === "all" || (f === "unreviewed" ? !it.reviewed : (it.split || "train") === f)));
  pos = Math.min(pos, Math.max(0, view.length - 1)); show();
}
function show() {
  const done = items.filter(i => i.reviewed).length;
  $("bar").style.width = (100 * done / Math.max(1, items.length)) + "%";
  $("count").textContent = `${done} / ${items.length} reviewed · ${view.length ? pos + 1 : 0} of ${view.length} shown`;
  if (!view.length) { cur = null; $("meta").innerHTML = "<span class='chip ok'>Nothing left in this view</span>";
    $("frame").removeAttribute("src"); $("context").textContent = ""; return; }
  cur = JSON.parse(JSON.stringify(view[pos]));
  $("frame").src = "/img/" + encodeURIComponent(cur.image);
  $("meta").innerHTML = [cur.source, `split: ${cur.split || "train"}`, `label: ${cur.label_source}`,
    `ego ${Math.round(cur.ego_speed_kmh)} km/h`, `cruise ${Math.round(cur.cruise_speed_kmh)} km/h`,
    ...(cur.flags || []).map(f => "⚑ " + f)]
    .map(t => `<span class="chip">${t}</span>`).join("") +
    (cur.excluded ? "<span class='chip warn'>excluded</span>" :
     cur.reviewed ? "<span class='chip ok'>reviewed</span>" : "<span class='chip warn'>auto label</span>");
  $("speed").value = Math.round(cur.target.target_speed_kmh ?? 0);
  $("risk").value = cur.target.risk || "low"; $("reason").value = cur.target.reason || "";
  $("context").textContent = cur.context; paint();
}
async function save(exclude = false) {
  if (!cur) return;
  cur.target.target_speed_kmh = Number($("speed").value); cur.target.risk = $("risk").value;
  cur.target.reason = $("reason").value.trim();
  const r = await fetch("/api/save", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: cur.id, target: cur.target, exclude})});
  if (!r.ok) { alert("Save failed: " + await r.text()); return; }
  const it = items.find(i => i.id === cur.id); it.target = cur.target; it.reviewed = true; it.excluded = exclude;
  $("toast").classList.add("show"); setTimeout(() => $("toast").classList.remove("show"), 500);
  if ($("filter").value === "unreviewed") { view.splice(pos, 1); if (pos >= view.length) pos = Math.max(0, view.length - 1); }
  else if (pos < view.length - 1) pos++;
  show();
}
document.addEventListener("keydown", e => {
  if (["TEXTAREA","INPUT","SELECT"].includes(document.activeElement.tagName)) {
    if (e.key === "Escape") document.activeElement.blur(); return; }
  const k = e.key.toLowerCase();
  if (LONG_KEYS.includes(k)) pick("longitudinal", LONG[LONG_KEYS.indexOf(k)]);
  else if (LAT_KEYS.includes(k)) pick("lateral", LAT[LAT_KEYS.indexOf(k)]);
  else if (e.key === "Enter") { e.preventDefault(); save(); }
  else if (k === "x") save(true);
  else if (e.key === "ArrowRight" && pos < view.length - 1) { pos++; show(); }
  else if (e.key === "ArrowLeft" && pos > 0) { pos--; show(); }
  else if (k === "r") { e.preventDefault(); $("reason").focus(); }
});
$("save").onclick = () => save(); $("exclude").onclick = () => save(true); $("next").onclick = () => { if (pos < view.length - 1) { pos++; show(); } };
$("prev").onclick = () => { if (pos > 0) { pos--; show(); } };
$("filter").onchange = () => { pos = 0; applyFilter(); }; $("source").onchange = () => { pos = 0; applyFilter(); };
opts($("long"), LONG, LONG_KEYS, "longitudinal"); opts($("lat"), LAT, LAT_KEYS, "lateral");
fetch("/api/items").then(r => r.json()).then(d => { items = d;
  [...new Set(d.map(i => i.source))].forEach(s => { const o = document.createElement("option"); o.value = o.textContent = s;
    $("source").appendChild(o); }); applyFilter(); });
</script></body></html>
"""

FIELDS = ("id", "image", "context", "target", "split", "source", "label_source", "reviewed",
          "ego_speed_kmh", "cruise_speed_kmh", "flags", "excluded")


def review_priority(rec: dict) -> tuple:
    """Val first (it decides the metrics), then flagged labels, then rarer actions, then the rest."""
    t = rec["target"]
    rare = t.get("longitudinal") != "KEEP" or t.get("lateral") != "KEEP_LANE"
    return (rec.get("split") != "val", not rec.get("flags"), not rare, rec["id"])


def serve(labels: str, port: int = 8765) -> None:
    labels_path = Path(labels).resolve()
    root = labels_path.parent
    def load() -> dict[str, dict]:
        return {r["id"]: {k: r.get(k) for k in FIELDS}
                for r in sorted(load_records(labels_path, include_excluded=True), key=review_priority)}

    records = load()  # reloaded on every page load, so builders can keep appending samples
    lock = threading.Lock()
    page = PAGE.replace("__LONG__", json.dumps(VLM_LONG_ACTIONS)).replace("__LAT__", json.dumps(VLM_LAT_ACTIONS))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/items":
                with lock:
                    records.clear()
                    records.update(load())
                    body = json.dumps(list(records.values()), ensure_ascii=False).encode()
                self._send(200, body, "application/json")
            elif url.path.startswith("/img/"):
                path = (root / unquote(url.path[len("/img/"):])).resolve()
                if root not in path.parents or not path.is_file():
                    self._send(404, b"not found", "text/plain")
                else:
                    self._send(200, path.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/save":
                self._send(404, b"not found", "text/plain")
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                sample_id, target, exclude = data["id"], data["target"], bool(data.get("exclude"))
                if sample_id not in records:
                    raise KeyError(f"unknown id {sample_id}")
                if target.get("longitudinal") not in VLM_LONG_ACTIONS or target.get("lateral") not in VLM_LAT_ACTIONS:
                    raise ValueError("invalid action")
                target = {"longitudinal": target["longitudinal"], "lateral": target["lateral"],
                          "target_speed_kmh": round(float(target["target_speed_kmh"])),
                          "risk": target.get("risk", "low"), "reason": str(target.get("reason", ""))[:200]}
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
                self._send(400, str(e).encode(), "text/plain")
                return
            with lock:
                with reviews_path(labels_path).open("a") as f:
                    f.write(json.dumps({"id": sample_id, "target": target, "exclude": exclude,
                                        "time": datetime.now().isoformat(timespec="seconds")},
                                       ensure_ascii=False) + "\n")
                records[sample_id].update(target=target, reviewed=True, excluded=exclude)
            self._send(200, b"ok", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    done = sum(r["reviewed"] for r in records.values())
    print(f"Reviewing {len(records)} samples ({done} already reviewed) from {labels_path}")
    print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop; reviews saved to {reviews_path(labels_path).name})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
