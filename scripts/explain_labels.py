#!/usr/bin/env python3
"""Richer VLM targets: a teacher VLM (Qwen2.5-VL-7B, 4-bit; configs/teacher_7b.yaml) writes, for labelled frames,
a short scene description, the hazards that matter and the reason for the decision ALREADY in the label (the
teacher is given the decision and never changes it). v3's own reasons are 43 template sentences (mean 7 words).

  python scripts/explain_labels.py generate --split train --keep 4400 --stop-at 07:30   # overnight (GPU)
  python scripts/explain_labels.py generate --split val --limit 300                     # to read / compare
  python scripts/explain_labels.py show --n 8                                           # a few, for review
  python scripts/explain_labels.py status                    # explained / accepted per split, rejection reasons
  python scripts/explain_labels.py rescore                   # re-apply the checks to the stored teacher outputs
  python scripts/explain_labels.py merge --out data/ds_v3x     # dataset for `adas-vla train --set vlm.explain=true`
  python scripts/explain_labels.py score --report R --data data/ds_v3x/val_explained.jsonl   # a student's output

generate appends to <labels>.explanations.jsonl and skips ids already there (resumable; --stop-at HH:MM ends at the
next sample, exit code 2 while samples remain). train selection: every sample that is not KEEP + `--keep` KEEP
samples (spread by a hash of the id), so the class balance of the v3 recipe (max class share 0.55) still holds.
merge writes a dataset (reviews and the split overlay already applied) whose train split holds only the explained
records, with scene / hazards / reason in their target, and whose val split is the full original val (the fixed
900 of the v3 numbers; explained ones carry the explanation too), plus val_explained.jsonl = the explained val
records alone (val loss of the training and the explanation check of the trained model).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data/ds_v3/labels.jsonl"
FIELDS = {"scene": 15, "hazards": 15, "reason": 20}  # max words of each field

TEACHER_SYSTEM = (
    "You annotate dashcam frames for a driver-assistance system. You get the front camera image, the perception "
    "summary and the correct driving decision. Explain the decision for the driver, briefly and concretely. Only "
    "mention what is visible in the image or listed in the perception data, and never contradict the decision."
)
TEACHER_USER = """Ego speed: {ego:.0f} km/h. Cruise set speed: {cruise:.0f} km/h.
Perception (monocular estimates):
{context}

Known facts (from the vehicle data, always true):
- Correct decision: longitudinal {long}, lateral {lat}, target speed {speed} km/h.
- Cause of the decision: {cause}{hint}
Write for the driver:
- scene: road type, traffic, light or weather (max 15 words)
- hazards: the objects that matter for this decision, the cause first, with positions and distances copied exactly from the perception list, or "none" (max 15 words)
- reason: why the decision is right, consistent with the cause above, positions as stated there (max 20 words)
Return JSON: {{"scene": "...", "hazards": "...", "reason": "..."}}"""
# a third of the non-KEEP train labels (Nexar crash clips) carry this cause, which does not say WHAT enters the path
GENERIC_CAUSE = "Hazard entering our path"
GENERIC_HINT = ("\n- The cause does not name the hazard: find it in the image. It is the road user that enters or blocks "
                "our lane: often one marked CUTTING IN or with the shortest TTC, else the nearest one in or next to "
                "our lane.")
RETRY = ("That answer was rejected: {why}. Write the JSON again: within the word limits, positions and distances "
         "exactly as in the perception list, consistent with the image and the cause.")
# a reason must not contradict its longitudinal action ("maintain a safe gap" is fine when slowing down, "maintain
# speed" is not; "avoid braking" is no contradiction of KEEP, "brake lights" never is)
_BRAKE = r"\bbrak(?:e|es|ing)\b(?! lights?)"
_KEEP_SPEED = r"\b(?:maintain|maintaining|keep|keeping|hold|holding)(?: the| its| our)?(?: current)? speed\b"
_FASTER = r"accelerat|\bspeed(?:s|ing)? up\b"
CONTRADICTS = {"KEEP": (_FASTER, _BRAKE), "ACCELERATE": (_BRAKE, r"\bslow(?:s|ing)? down\b", _KEEP_SPEED),
               **{a: (_FASTER, _KEEP_SPEED) for a in ("DECELERATE", "BRAKE", "EMERGENCY_BRAKE", "STOP")}}
_NEGATED = (r"\b(?:avoid\w*|prevent\w*|minimi[sz]\w*|reduc\w*|without|no need to|no|not|never|unnecessary|instead of"
            r"|rather than)\b(?:\s+\w+){0,3}\s*$")
# the road user / light a cause names must be in the explanation (a retry once explained "car running across the
# intersection" with the traffic light it was told to place right)
CAUSE_OBJECTS = {"car": r"\b(?:cars?|vehicles?|sedans?|suvs?)\b", "truck": r"\b(?:trucks?|lorr(?:y|ies)|vehicles?)\b",
                 "ute": r"\b(?:utes?|pickups?|trucks?|vehicles?)\b", "bus": r"\b(?:bus(?:es)?|vehicles?)\b",
                 "van": r"\b(?:vans?|vehicles?)\b", "motorcycle": r"\b(?:motorcycles?|motorbikes?|bikes?|riders?)\b",
                 "pedestrian": r"\b(?:pedestrians?|people|person|walkers?)\b", "cyclist": r"\b(?:cyclists?|bicycles?|bikes?)\b",
                 "light": r"\blights?\b", "traffic": r"\b(?:traffic|cars?|vehicles?)\b"}
# the explanation is read by the driver: no words of the annotation prompt
META = (r"\b(?:low|medium|high)[- ]risk\b|\brisk (?:condition|level|rating)\b|\bperception (?:list|data|summary)\b"
        r"|\bknown facts?\b|\bcorrect decision\b|\bthe cause\b|\bmonocular\b")
NO_TRAFFIC = (r"\bno (?:other )?(?:traffic|vehicles?|cars?|road users)\b"
              r"(?!\s+(?:in (?:our|the|this|my|ego|same) lane|ahead|directly|close|in front))"
              r"|\bempty (?:road|highway|street|lane)s?\b|\bdeserted\b")
VEHICLE = r"\b(?:cars?|trucks?|bus(?:es)?|vehicles?|motorcycles?|motorbikes?|vans?|utes?|brake lights?)\b"
CLEAR = (r"\b(?:road|lane|path|way)(?: ahead)? (?:is|remains) clear\b|\bclear (?:road|lane|path)\b(?! mark)"
         r"|\bclear ahead\b|\bnothing ahead\b")
AHEAD = r"\bahead\b|\bin front\b|\b(?:same|our|ego) lane\b"
LEAD = r"\b(?:lead|cutting in|pedestrians?|cyclists?)\b|\b(?:car|truck|bus|vehicle|motorcycle)s? (?:ahead|in front)\b"
# the position words of a claim: the compound ones first ("front-left" is no "left" + "in front")
_POSITIONS = (("front-left", r"\bfront[- ]left\b|\bleft[- ]front\b"), ("front-right", r"\bfront[- ]right\b|\bright[- ]front\b"),
              ("ahead", AHEAD), ("front-left", r"\bleft\b"), ("front-right", r"\bright\b"))
_DISTANCE = r"(\d+(?:\.\d+)?)\s?(?:m|meters?|metres?)\b"
# a road user entering our lane is "in front" whatever side the detector put it on
ENTERING = r"\bcut(?:s|ting)? (?:in|into)\b|\benter(?:s|ing)?\b|\bmerg(?:e|es|ing)\b|\bpull(?:s|ing)? (?:in|into)\b"
_CONJUNCTIONS = ("and", "or", "but", "while", "which", "as", "so")
_OBJECT = r"^- ([^,]+), (ahead|front-left|front-right), ~(\d+(?:\.\d+)?) m(.*)$"
VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")
PREFIX = '{"scene": "'


def explanations_path(labels: Path) -> Path:
    return labels.with_name(labels.stem + ".explanations.jsonl")


def teacher_rows(labels: Path) -> list[dict]:
    path = explanations_path(labels)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def load_explanations(labels: Path) -> dict[str, dict]:
    return {row["id"]: row for row in teacher_rows(labels) if row.get("ok")}


def perception_objects(context: str) -> list[tuple[str, str, float, str]]:
    """(class, position, distance m, flags) of every object of the perception list."""
    return [(m[1], m[2], float(m[3]), m[4]) for m in re.finditer(_OBJECT, context, re.M)]


def _positions(clause: str) -> set[str]:
    found = set()
    for position, pattern in _POSITIONS:
        if re.search(pattern, clause, re.I):
            found.add(position)
            clause = re.sub(pattern, " ", clause, flags=re.I)
    return found


def distance_claims(text: str) -> list[tuple[set[str], float, bool]]:
    """(positions, distance, entering our lane) of every distance in the text. The positions come from its clause,
    else from the previous clause of the same sentence: "Car ahead, 37 meters" -> ({"ahead"}, 37.0, False)."""
    claims = []
    for sentence in re.split(r";|\.(?:\s|$)", text):
        last: set[str] = set()
        entering = re.search(ENTERING, sentence, re.I) is not None
        for clause in sentence.split(","):
            positions = _positions(clause) or last
            claims += [(positions, float(d), entering) for d in re.findall(_DISTANCE, clause, re.I)]
            last = positions
    return claims


def _fits(positions: set[str], distance: float, entering: bool, obj: tuple[str, str, float, str]) -> bool:
    _, position, d, flags = obj
    if abs(d - distance) > 1.0:
        return False
    return not positions or position in positions or ("ahead" in positions and ("IN EGO LANE" in flags or entering))


def shorten(text: str, limit: int) -> str | None:
    """A field cut to its word limit at the end of a sentence, else of a ';' item or a ',' clause (the dropped part
    is the least important tail: '..., ensuring a safe gap'); None when no cut keeps 5 words or more."""
    words = text.split()
    if len(words) <= limit:
        return text
    for ends in ((".", ";"), (",",)):
        for i in range(limit - 1, 3, -1):
            if not words[i].endswith(ends):
                continue
            start = max((j + 1 for j in range(i) if words[j].endswith((",", ";", "."))), default=0)
            if words[start].lower() in _CONJUNCTIONS and i - start < 3:  # "..., and another car," would dangle
                continue
            return " ".join(words[:i + 1]).rstrip(",;") + ("" if words[i].endswith(".") else ".")
    return None


def drop_meta_tail(text: str) -> str:
    """'..., ensuring a low-risk driving condition.' -> '...' (prompt words in a trailing clause)."""
    m = re.search(META, text, re.I)
    cut = text.rfind(", ", 0, m.start()) if m else -1
    return text[:cut] + "." if cut > 0 and len(text[:cut].split()) >= 5 else text


def driver_words(text: str) -> str:
    """The perception list's words for our own car / lane, as a driver says them."""
    text = re.sub(r"\bIN EGO LANE\b", "in our lane", re.sub(r"\bCUTTING IN\b", "cutting in", text))
    text = re.sub(r"\b(?:the )?ego(?:'s)? lane\b", "our lane", text, flags=re.I)
    text = re.sub(r"\b(?:the )?ego(?:'s)? (?:vehicle|car)\b", "our car", text, flags=re.I)
    return text[:1].upper() + text[1:]


def clean(raw: str, rec: dict) -> tuple[dict | None, str]:
    """The teacher's JSON -> ({scene, hazards, reason} within the word limits, "") or (None, why it was rejected):
    every distance it quotes must be an object of the perception list at that position (+-1 m), a traffic light
    colour must be one the detector saw, the fields must not contradict each other ("no traffic" next to a car,
    "lane is clear" next to a car ahead) nor the action, and they must not echo the annotation prompt."""
    from adas_vla.reasoning.parser import extract_json

    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return None, "no json"
    out = {}
    for key, limit in FIELDS.items():
        text = obj.get(key)
        if not isinstance(text, str) or not text.strip():
            return None, f"no {key}"
        text = shorten(drop_meta_tail(driver_words(" ".join(text.split()))), limit)
        if text is None:
            return None, f"{key} longer than {limit} words"
        out[key] = text.rstrip(",;:")
    if re.fullmatch(r"(?:none|no hazards?|nothing)(?: detected| identified| visible)?\.?", out["hazards"], re.I):
        out["hazards"] = "none"
    objects = perception_objects(rec["context"])
    for positions, distance, entering in distance_claims(out["hazards"] + ". " + out["reason"]):
        fits = [o for o in objects if _fits(positions, distance, entering, o)]
        if entering and distance > 30 and fits and all(o[1] == "ahead" and "CUTTING IN" not in o[3] for o in fits):
            return None, f"the {fits[0][0]} at {fits[0][2]:g} m is already ahead in our lane, it is not entering it"
        if not fits:
            near = [o for o in objects if abs(o[2] - distance) <= 1.0]
            if near:  # told to the teacher on its second try
                return None, (f"the {near[0][0]} at {near[0][2]:g} m is {near[0][1]} in the perception list, not "
                              f"{'/'.join(sorted(positions))}")
            return None, f"no object at {distance:g} m in the perception list"
    speeds = [rec["ego_speed_kmh"], rec["cruise_speed_kmh"], rec["target"]["target_speed_kmh"],
              *map(float, re.findall(r"(\d+(?:\.\d+)?) km/h", rec["context"]))]
    for v in re.findall(r"(\d+(?:\.\d+)?)\s?km/h", out["hazards"] + " " + out["reason"]):
        if not any(abs(float(v) - k) <= 2.0 for k in speeds):
            return None, f"{v} km/h is neither our speed, the cruise / target speed nor a speed of the perception list"
    # traffic-light colours only as the detector saw them (the teacher read red brake lights as a red light)
    text = " ".join(out.values()).lower()
    seen = set(re.findall(r"traffic light \((\w+)\)", rec["context"]))
    for colour in ("red", "green", "yellow", "amber"):
        if re.search(rf"\b{colour}\b[^.;]*\blights?\b|\blights?\b[^.;]*\b{colour}\b", text) and \
                (colour if colour != "amber" else "yellow") not in seen and "brake light" not in text:
            return None, f"{colour} light not in the perception list"
    if re.search(META, text):
        return None, f"prompt words for the driver: {re.search(META, text)[0]}"
    hazards = out["hazards"].lower()
    if re.search(NO_TRAFFIC, text):
        near = [o for o in objects if o[0] in VEHICLE_CLASSES and o[2] <= 30]
        if (hazards != "none" and re.search(VEHICLE, hazards)) or near:
            return None, "says there is no traffic but " + (f"a {near[0][0]} is at {near[0][2]:g} m" if near
                                                            else "lists vehicles")
    reason = out["reason"].lower()
    if re.search(CLEAR, reason) and hazards != "none" and re.search(AHEAD, hazards):
        return None, "says the lane is clear but lists an object ahead"
    if hazards == "none" and re.search(LEAD, reason):
        return None, "hazards none but the reason names a road user"
    cause = str(rec["target"].get("reason", "")).lower()
    named = [k for k in CAUSE_OBJECTS if re.search(rf"\b{k}s?\b", cause)]
    if named and not any(re.search(CAUSE_OBJECTS[k], hazards + " " + reason) for k in named):
        return None, f"the explanation does not mention the {named[0]} of the cause"
    action = rec["target"]["longitudinal"]
    for pattern in CONTRADICTS.get(action, ()):
        for m in re.finditer(pattern, reason):
            if not re.search(_NEGATED, reason[max(0, m.start() - 40):m.start()]):
                return None, f"reason contradicts {action}: {m[0]}"
    return out, ""


def teacher_messages(rec: dict, cfg) -> list[dict]:
    from adas_vla.training.data import sample_visual

    t = rec["target"]
    cause = t.get("reason", "-")
    user = TEACHER_USER.format(ego=rec["ego_speed_kmh"], cruise=rec["cruise_speed_kmh"], context=rec["context"],
                               long=t["longitudinal"], lat=t["lateral"], speed=t["target_speed_kmh"], cause=cause, hint=GENERIC_HINT if cause.startswith(GENERIC_CAUSE) else "")
    return [{"role": "system", "content": [{"type": "text", "text": TEACHER_SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": sample_visual(rec, cfg)},
                                         {"type": "text", "text": user}]}]


def select(records: list[dict], split: str, keep: int | None, limit: int | None) -> list[dict]:
    recs = [r for r in records if r.get("split", "train") == split]
    if split == "train" and keep is not None:
        h = lambda r: hashlib.sha1(r["id"].encode()).hexdigest()  # noqa: E731
        keeps = sorted((r for r in recs if r["target"]["longitudinal"] == "KEEP"), key=h)[:keep]
        recs = [r for r in recs if r["target"]["longitudinal"] != "KEEP"] + keeps
    if limit:
        step = max(1, len(recs) // limit)
        recs = recs[::step][:limit]
    return recs


def cmd_generate(args) -> None:
    from adas_vla.config import load_config
    from adas_vla.reasoning.vlm import VisionLanguageModel
    from adas_vla.training.data import load_records

    labels = Path(args.data)
    done = {row["id"] for row in teacher_rows(labels)}
    todo = [r for r in select(load_records(labels), args.split, args.keep, args.limit) if r["id"] not in done]
    print(f"{len(todo)} {args.split} samples to explain ({len(done)} already in {explanations_path(labels).name})",
          flush=True)
    stop = None
    if args.stop_at:
        h, m = map(int, args.stop_at.split(":"))
        now = dt.datetime.now()
        stop = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if stop <= now:
            stop += dt.timedelta(days=1)
    cfg = load_config(args.config)
    vlm = VisionLanguageModel(cfg.vlm)
    t0, n, rejected, retried = time.time(), 0, 0, 0
    with explanations_path(labels).open("a") as f:
        for rec in todo:
            if stop is not None and dt.datetime.now() >= stop:
                print(f"stop-at {args.stop_at} reached: {len(todo) - n} {args.split} samples left", flush=True)
                sys.exit(2)
            messages = teacher_messages(rec, cfg)
            raw = vlm.generate(messages, max_new_tokens=args.max_new_tokens, prefix=PREFIX, stop_strings=['"}'])
            fields, why = clean(raw, rec)
            first = why
            for _ in range(args.retries if fields is None else 0):  # once more, told why the answer was rejected
                messages += [{"role": "assistant", "content": [{"type": "text", "text": raw}]},
                             {"role": "user", "content": [{"type": "text", "text": RETRY.format(why=why)}]}]
                raw = vlm.generate(messages, max_new_tokens=args.max_new_tokens, prefix=PREFIX, stop_strings=['"}'])
                fields, why = clean(raw, rec)
                retried += 1
                if fields is not None:
                    break
            f.write(json.dumps({"id": rec["id"], "ok": fields is not None, **(fields or {}), "rejected": why,
                                "first_rejected": first, "teacher": cfg.vlm.model_id, "raw": raw},
                               ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            rejected += fields is None
            if n % 25 == 0 or n <= 3:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(todo)} {rate:.1f} s/sample, ETA {(len(todo) - n) * rate / 3600:.1f} h, retried "
                      f"{retried / n:.0%}, rejected {rejected / n:.0%} | {rec['id']}: {fields or why}", flush=True)


def cmd_show(args) -> None:
    from adas_vla.training.data import load_records

    labels = Path(args.data)
    expl = load_explanations(labels)
    recs = {r["id"]: r for r in load_records(labels)}
    for i, (rid, e) in enumerate(list(expl.items())[:args.n]):
        r = recs[rid]
        print(f"[{rid}] {r['target']['longitudinal']}/{r['target']['lateral']} | {r['context'].splitlines()[-1][:90]}")
        print(f"   old reason: {r['target'].get('reason')}")
        print(f"   scene  : {e['scene']}\n   hazards: {e['hazards']}\n   reason : {e['reason']}")


def _why(row: dict) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "N", row["rejected"])  # "distance 12 m not in ..." -> one bucket


def cmd_status(args) -> None:
    """Explained / accepted per split and action. Exit code 1 while samples are left, 3 when the acceptance is below
    --min-accept (a split) or 0.15 below it (an action with 20+ samples): something to look at before training."""
    from adas_vla.training.data import load_records

    labels = Path(args.data)
    rows = {row["id"]: row for row in teacher_rows(labels)}
    records = load_records(labels)
    left, low = 0, []
    for split, keep, limit in (("val", None, args.val), ("train", args.keep, None)):
        selection = select(records, split, keep, limit)
        done = [(r["target"]["longitudinal"], rows[r["id"]]) for r in selection if r["id"] in rows]
        ok = sum(row["ok"] for _, row in done)
        print(f"{split}: {len(done)}/{len(selection)} explained, {ok} accepted ({ok / max(1, len(done)):.0%})")
        if args.min_accept and done and ok / len(done) < args.min_accept:
            low.append(split)
        for action in sorted({a for a, _ in done}):
            n = sum(a == action for a, _ in done)
            n_ok = sum(row["ok"] for a, row in done if a == action)
            print(f"   {action:16s} {n_ok:5d}/{n:<5d} accepted ({n_ok / n:.0%})")
            if args.min_accept and n >= 20 and n_ok / n < args.min_accept - 0.15:
                low.append(f"{split} {action}")
        for why, n in Counter(_why(row) for _, row in done if not row["ok"]).most_common(6):
            print(f"   rejected {n:5d}  {why}")
        left += len(selection) - len(done)
    if left:
        sys.exit(1)
    if low:
        print(f"acceptance below {args.min_accept:.0%}: {', '.join(low)}")
        sys.exit(3)


def cmd_rescore(args) -> None:
    """Re-apply clean() to every stored teacher output (after a change of the checks), in place."""
    from adas_vla.training.data import load_records

    labels = Path(args.data)
    recs = {r["id"]: r for r in load_records(labels)}
    rows, flips = [], Counter()
    for row in teacher_rows(labels):
        if row["id"] in recs:  # (a record excluded since keeps its row, unused)
            fields, why = clean(row["raw"], recs[row["id"]])
            flips[(row["ok"], fields is not None)] += 1
            row = {"id": row["id"], "ok": fields is not None, **(fields or {}), "rejected": why,
                   "first_rejected": row.get("first_rejected", ""), "teacher": row["teacher"], "raw": row["raw"]}
        rows.append(row)
    path = explanations_path(labels)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    os.replace(tmp, path)
    print(f"{path.name}: {len(rows)} rows; accepted -> rejected {flips[(True, False)]}, rejected -> accepted "
          f"{flips[(False, True)]}, accepted {sum(row['ok'] for row in rows)}")


def cmd_merge(args) -> None:
    from adas_vla.training.data import load_records

    labels = Path(args.data)
    expl = load_explanations(labels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = Path("..") / labels.parent.name  # image paths stay relative to the new labels file
    n_train = n_val = n_val_expl = 0
    with (out / "labels.jsonl").open("w") as f, (out / "val_explained.jsonl").open("w") as fv:
        for rec in load_records(labels):
            split = rec.get("split", "train")
            if split == "train" and rec["id"] not in expl:
                continue
            rec = {k: v for k, v in rec.items() if k not in ("image_path", "image_prev_path", "reviewed", "excluded")}
            rec["image"] = str(frames / rec["image"])
            if rec.get("image_prev"):
                rec["image_prev"] = str(frames / rec["image_prev"])
            if rec["id"] in expl:
                e = expl[rec["id"]]
                rec["target"] = {**{k: v for k, v in rec["target"].items() if k != "reason"},
                                 "scene": e["scene"], "hazards": e["hazards"], "reason": e["reason"]}
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            f.write(line)
            if split == "val" and rec["id"] in expl:
                fv.write(line)
                n_val_expl += 1
            n_train += split == "train"
            n_val += split == "val"
    print(f"{out / 'labels.jsonl'}: {n_train} explained train records, {n_val} val records "
          f"({n_val_expl} explained, also in val_explained.jsonl)")


def cmd_score(args) -> None:
    """The explanations in an `adas-vla eval --report` (run with vlm.explain=true + vlm.generate_reason=true on
    val_explained.jsonl) under the teacher's checks (its own action for the contradiction check), next to the
    teacher's explanations of the same frames."""
    from adas_vla.training.data import load_records

    recs = {r["image"]: r for r in load_records(args.data)}
    rows = [json.loads(line) for line in Path(args.report).read_text().splitlines()]
    print(f"{len(rows)} frames of {args.report}")
    print(f"{'':28s}{'model':>10s}{'teacher':>10s}")
    stats = {}
    for who in ("model", "teacher"):
        n, ok, why = 0, 0, Counter()
        words = {k: [] for k in FIELDS}
        reasons = set()
        for row in rows:
            out = row["pred"] if who == "model" else recs[row["image"]]["target"]
            n += 1
            if not out:
                why["invalid JSON"] += 1
                continue
            for k in FIELDS:
                if out.get(k):
                    words[k].append(len(out[k].split()))
            reasons.add(" ".join(str(out.get("reason", "")).lower().split()))
            if not all(out.get(k) for k in FIELDS):
                why["missing scene / hazards / reason"] += 1
                continue
            fields, reason = clean(json.dumps(out), {**recs[row["image"]], "target": out})
            ok += fields is not None
            if fields is None:
                why[_why({"rejected": reason})] += 1
        stats[who] = (n, ok, words, reasons, why)
    m, t = stats["model"], stats["teacher"]
    print(f"{'passes the checks':28s}{m[1] / max(1, m[0]):>10.0%}{t[1] / max(1, t[0]):>10.0%}")
    def mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    for k in FIELDS:
        print(f"{'mean words ' + k:28s}{mean(m[2][k]):>10.1f}{mean(t[2][k]):>10.1f}")
    print(f"{'distinct reasons':28s}{len(m[3]):>10d}{len(t[3]):>10d}")
    for why, n in m[4].most_common(8):
        print(f"   model rejected {n:4d}  {why}")
    for row in rows[:args.examples]:
        rec, p = recs[row["image"]], row["pred"] or {}
        print(f"\n[{rec['id']}] label {row['gt']['longitudinal']}, model {p.get('longitudinal')}")
        for k in FIELDS:
            print(f"   {k:7s} model  : {p.get(k, '-')}\n   {'':7s} teacher: {rec['target'].get(k, '-')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--data", default=str(DATA))
    g.add_argument("--split", default="train", choices=["train", "val"])
    g.add_argument("--keep", type=int, default=4400, help="KEEP samples to explain (train), next to all the others")
    g.add_argument("--limit", type=int, help="at most this many samples (spread over the split)")
    g.add_argument("--config", default=str(ROOT / "configs/teacher_7b.yaml"))
    g.add_argument("--max-new-tokens", type=int, default=110)
    g.add_argument("--retries", type=int, default=1, help="new answers after a rejection, told the reason")
    g.add_argument("--stop-at", help="HH:MM: stop at the next sample after this time (resumable)")
    g.set_defaults(func=cmd_generate)
    s = sub.add_parser("show")
    s.add_argument("--data", default=str(DATA))
    s.add_argument("--n", type=int, default=8)
    s.set_defaults(func=cmd_show)
    st = sub.add_parser("status", help="exit code 1 while samples of the selection are left")
    st.add_argument("--data", default=str(DATA))
    st.add_argument("--keep", type=int, default=4400)
    st.add_argument("--val", type=int, default=300, help="val samples of the selection (generate --limit)")
    st.add_argument("--min-accept", type=float, default=0.0, help="e.g. 0.85: exit code 3 below it")
    st.set_defaults(func=cmd_status)
    r = sub.add_parser("rescore")
    r.add_argument("--data", default=str(DATA))
    r.set_defaults(func=cmd_rescore)
    m = sub.add_parser("merge")
    m.add_argument("--data", default=str(DATA))
    m.add_argument("--out", default=str(ROOT / "data/ds_v3x"))
    m.set_defaults(func=cmd_merge)
    sc = sub.add_parser("score")
    sc.add_argument("--report", required=True, help="adas-vla eval --report output")
    sc.add_argument("--data", default=str(ROOT / "data/ds_v3x/val_explained.jsonl"), help="the eval's --data")
    sc.add_argument("--examples", type=int, default=6)
    sc.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
