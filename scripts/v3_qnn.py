#!/usr/bin/env python3
"""adas-vlm-v3 (Qwen2.5-VL-3B fine-tuned) -> Qualcomm HTP: reference, checks, export (see adas_vla/deploy/qwen_vl.py).

  check-hf      transformers model, float32 on the CPU, on N val samples: input ids, pixel values, M-RoPE
                positions, image features, last-prompt-token logits, greedy tokens -> <out>/check_hf.pt
  check-deploy  the same samples through the deployment form (patchify, ViT with window masks, 4 LLM parts in
                Genie layout with AR128 prefill + AR1 decode, LUT embedding) and compare with check_hf.pt
  eval          the val split through the deployment form on the GPU (bf16), float or with simulated
                quantization (--rotate --wbits 4 [--wblock 64] --kv-bits 8): metrics like `adas-vla eval`
  vit-cache     float32 ViT features of the val images (GPU) for eval --vit-cache
  export        float32 ONNX for QAIRT: vit, part1..part4 (AR128 + AR1 graphs, CL 1024), lut (host embedding table),
                Hadamard-rotated by default; one part at a time (~9 GB of RAM)
  calib         calibration inputs for qairt-quantizer (AR128 chunks + AR1 steps per part, ViT patches) captured
                from the float model on train prompts, and stats.json (ranges shared between graphs)
  calib-vit     ViT calibration frames (train) and evaluation frames (val), preprocessing only
  check-bins    KV / hidden-state encodings agree between the graphs of the context binaries (info first)
  eval-part     one quantized LLM graph on the x86 HTP emulator vs float ONNX: SQNR of outputs, logits top-1
  eval-vit      quantized ViT on the x86 HTP emulator vs float (SQNR, token cosine) on the val frames
  gptq          GPTQ 4-bit LLM weights (per channel or LPBQ) from train prompts, layer by layer on the GPU
  eval-board    the val split through v3 on the board (MODEL=v3 board/vlm_server.sh start): metrics + timings
  compare       eval reports on their common samples: agreement with the first and joint / under-braking of each
  bundle        Genie bundle for the board (context binaries, LUT, tokenizer, ViT constants, configs) per SoC
  overrides     encoding overrides of the LLM graphs from stats.json (scripts/v3_qnn_build.sh convert uses them)

Both run on the CPU in float32 (about 16 GB of RAM each, one after the other):
  systemd-run --user --scope -p MemoryMax=24G .venv/bin/python scripts/v3_qnn.py check-hf
  systemd-run --user --scope -p MemoryMax=24G .venv/bin/python scripts/v3_qnn.py check-deploy
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adas_vla.deploy import qwen_vl as qv  # noqa: E402
from adas_vla.reasoning.prompts import DECISION_PREFIX, decision_messages  # noqa: E402

MODEL = ROOT / "models/adas-vlm-v3"
DATA = ROOT / "data/ds_v3/labels.jsonl"
QNN_DIR = (ROOT / "data").resolve() / "qnn_v3"  # on the SSD (~30 GB of float32 ONNX); qairt.sh needs WORK=<its parent>
IMAGE_SPLIT = "<|vision_start|><|image_pad|><|vision_end|>"


def val_records(n: int, stride: int = 1) -> list[dict]:
    from adas_vla.training.evaluate import select_records

    return select_records(str(DATA), split="val")[::stride][:n]


def messages_of(rec: dict, image) -> list[dict]:
    return decision_messages(image, rec["context"], rec["ego_speed_kmh"], rec["cruise_speed_kmh"])


def pil_of(rec: dict):
    from PIL import Image

    from adas_vla.reasoning.vlm import to_pil

    return to_pil(Image.open(rec["image_path"]), 640, [560, 308])


def prompt_parts(processor, rec: dict, image) -> tuple[str, str]:
    """The decision prompt split around the image (what the board receives), with DECISION_PREFIX pre-filled."""
    text = processor.apply_chat_template(messages_of(rec, image), tokenize=False, add_generation_prompt=True)
    i = text.index(IMAGE_SPLIT)
    return text[:i] + "<|vision_start|>", text[i + len(IMAGE_SPLIT) - len("<|vision_end|>"):] + DECISION_PREFIX


def cmd_check_hf(args) -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from adas_vla.reasoning.vlm import append_tokens, encode_messages

    torch.set_num_threads(args.threads)
    processor = AutoProcessor.from_pretrained(MODEL)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu",
                                                               attn_implementation="eager").eval()
    print(f"HF model loaded in {time.time() - t0:.0f} s")
    out = []
    for rec in val_records(args.n, args.stride):
        image = pil_of(rec)
        inputs = encode_messages(processor, messages_of(rec, image), add_generation_prompt=True)
        inputs = append_tokens(inputs, processor.tokenizer(DECISION_PREFIX, add_special_tokens=False,
                                                           return_tensors="pt")["input_ids"])
        t0 = time.time()
        with torch.no_grad():
            feats = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
            feats = torch.cat(list(feats), 0) if isinstance(feats, (list, tuple)) else feats
            pos, _ = model.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"],
                                                inputs["image_grid_thw"])
            logits = model(**inputs).logits[0, -1]
            # plain greedy like Genie (generation_config.json would add repetition_penalty 1.05)
            gen = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, repetition_penalty=1.0)[0, inputs["input_ids"].shape[1]:]
        print(f"{rec['id']}: {time.time() - t0:.0f} s  {processor.tokenizer.decode(gen)!r}")
        out.append({"id": rec["id"], "input_ids": inputs["input_ids"][0], "pixel_values": inputs["pixel_values"],
                    "positions": pos[:, 0], "image_features": feats.float().clone(), "logits": logits.float().clone(),
                    "tokens": gen})
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save(out, Path(args.out) / "check_hf.pt")


def cmd_check_deploy(args) -> None:
    from tokenizers import Tokenizer
    from transformers import AutoProcessor

    torch.set_num_threads(args.threads)
    if args.mask is not None:
        qv.LLM_MASK = args.mask
    ref = {r["id"]: r for r in torch.load(Path(args.out) / "check_hf.pt")}
    processor = AutoProcessor.from_pretrained(MODEL)  # chat template only, as BoardVLM
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    spec = qv.Spec.from_model(MODEL)
    ck = qv.Checkpoint(MODEL)
    t0 = time.time()
    vit = qv.VisionEncoder(ck, spec).eval()
    parts = [qv.DecoderPart(ck, spec, r, last=i == 3).eval() for i, r in enumerate(qv.part_ranges(spec))]
    lut = qv.embedding_table(ck)
    gen = qv.Generator(parts, spec)
    vit_in = qv.vit_inputs(spec)
    print(f"deployment form built in {time.time() - t0:.0f} s; parts {[list(r)[0] for r in qv.part_ranges(spec)]}")
    report = []
    for rec in val_records(args.n, args.stride):
        r = ref[rec["id"]]
        image = pil_of(rec)
        before, after = prompt_parts(processor, rec, image)
        ids_b, ids_a = tok.encode(before, add_special_tokens=False).ids, tok.encode(after, add_special_tokens=False).ids
        ids = torch.tensor(ids_b + [spec.image_token] * spec.image_tokens + ids_a)
        pix = torch.from_numpy(qv.patchify(qv.resize_rgb(image, spec), spec))
        pos = qv.prompt_positions(len(ids_b), len(ids_a), spec)
        t0 = time.time()
        with torch.no_grad():
            feats = vit(pix, *(vit_in[k] for k in qv.VisionEncoder.INPUTS[1:]))
            embeds = torch.cat([lut[ids_b], feats, lut[ids_a]])
            gen.reset()
            logits = gen.prefill(embeds, pos)
            tokens = [int(logits.argmax())]
            while len(tokens) < len(r["tokens"]):
                tokens.append(int(gen.decode(lut[tokens[-1]]).argmax()))
        cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))  # noqa: E731
        row = {
            "id": rec["id"], "seconds": round(time.time() - t0, 1),
            "input_ids_equal": bool(torch.equal(ids, r["input_ids"])), "prompt_tokens": len(ids),
            "pixel_max_abs_diff": float((pix - r["pixel_values"]).abs().max()),
            "positions_equal": bool(torch.equal(pos, r["positions"])),
            "image_features_cos": cos(feats, r["image_features"]),
            "image_features_max_abs_diff": float((feats - r["image_features"]).abs().max()),
            "image_features_max_abs": float(r["image_features"].abs().max()),
            "logits_cos": cos(logits, r["logits"]),
            "logits_max_abs_diff": float((logits - r["logits"]).abs().max()),
            "top1_equal": int(logits.argmax()) == int(r["logits"].argmax()),
            "tokens_equal": tokens == r["tokens"].tolist(),
            "text": processor.tokenizer.decode(tokens),
        }
        print(json.dumps(row))
        report.append(row)
    tag = f"_mask{-args.mask:g}" if args.mask is not None else ""
    (Path(args.out) / f"check_deploy{tag}.json").write_text(json.dumps(report, indent=1))


def onnx_export(module, args, path: Path, inputs: list[str], outputs: list[str]) -> None:
    """Legacy (TorchScript) ONNX export, opset 17, weights in one external file <name>.onnx.data."""
    import shutil
    import tempfile

    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=path.parent))
    try:
        with torch.no_grad():
            torch.onnx.export(module, args, str(tmp / path.name), input_names=inputs, output_names=outputs,
                              opset_version=17, dynamo=False)
        model = onnx.load(str(tmp / path.name))
        for f in (path, path.with_name(path.name + ".data")):
            f.unlink(missing_ok=True)
        onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                  location=path.name + ".data")
    finally:
        shutil.rmtree(tmp)
    size = sum(f.stat().st_size for f in path.parent.glob(path.name + "*"))
    print(f"wrote {path} ({size / 2**30:.2f} GiB)", flush=True)


def part_sample(spec: qv.Spec, layers: int, ar: int, cl: int) -> tuple:
    hd, kv = spec.head_dim, spec.kv_heads
    past = [t for _ in range(layers) for t in (torch.zeros(kv, 1, hd, cl - ar), torch.zeros(kv, 1, cl - ar, hd))]
    return (torch.zeros(1, ar, spec.dim), torch.zeros(1, 1, ar, cl), torch.ones(1, 1, ar, hd // 2),
            torch.zeros(1, 1, ar, hd // 2), *past)


def graph_name(i: int, ar: int, cl: int, parts: int = 4) -> str:
    return f"ar{ar}_cl{cl}_{i + 1}_of_{parts}"


def cmd_export(args) -> None:
    """ONNX graphs (float32) for QAIRT: vision_encoder and the LLM parts as AR128 + AR1 graphs, and the host LUT."""
    spec, ck = qv.Spec.from_model(MODEL), qv.Checkpoint(MODEL)
    rot = qv.hadamard(spec.dim) if args.rotate else None
    head_rot = qv.hadamard(spec.head_dim, seed=1) if args.rotate_heads else None
    vit_rot = qv.orthogonal(spec.vit_dim) if args.rotate_vit else None
    qw = None
    if args.gptq:  # GPTQ weights (on their 4-bit grid) instead of the float projections
        from adas_vla.deploy.gptq import QuantizedWeights

        qw = QuantizedWeights(args.gptq)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"rotate": args.rotate, "rotate_heads": args.rotate_heads, "rotate_vit": args.rotate_vit,
            "hadamard_seed": 0, "cl": args.cl, "ar": args.ar, "image": [spec.image_w, spec.image_h]}
    if (out / "export.json").exists():  # keep what the other parts were exported with
        meta = {**json.loads((out / "export.json").read_text()), **meta}
    if args.gptq and any(p.startswith("part") for p in args.what):
        meta["gptq"] = str(args.gptq)
    (out / "export.json").write_text(json.dumps(meta))
    what = set(args.what)
    if "vit" in what:
        vit = qv.VisionEncoder(ck, spec, rot, vit_rotation=vit_rot).eval()
        ins = qv.vit_inputs(spec)
        sample = (torch.zeros(spec.patches, spec.patch_dim), *(ins[k] for k in qv.VisionEncoder.INPUTS[1:]))
        onnx_export(vit, sample, out / "vision_encoder" / "vision_encoder.onnx", list(qv.VisionEncoder.INPUTS),
                    ["image_features"])
        del vit
        gc.collect()
    ranges = qv.part_ranges(spec)
    for i, r in enumerate(ranges):
        if f"part{i + 1}" not in what:
            continue
        part = qv.DecoderPart(ck, spec, r, last=i == len(ranges) - 1, rotation=rot, head_rotation=head_rot,
                              quantized=qw).eval()
        ins, outs = part.io_names()
        for ar in (args.ar, 1):
            name = graph_name(i, ar, args.cl)
            onnx_export(part, part_sample(spec, len(r), ar, args.cl), out / name / f"{name}.onnx", ins, outs)
        del part
        gc.collect()
    if "lut" in what:
        qv.embedding_table(ck, rot).numpy().tofile(out / "embedding_weights.raw")
        print("wrote", out / "embedding_weights.raw")


def cmd_calib(args) -> None:
    """Calibration inputs of the LLM graphs for qairt-quantizer from the deployment form (rotated, and with the GPTQ
    weights when export.json names them) on train prompts: every AR128 prefill chunk and a few decode steps (AR1)
    per prompt for each part (the ViT's come from calib-vit). stats.json holds the ranges of the tensors whose
    encodings must be shared between graphs (KV cache per layer, hidden states between parts, inputs_embeds)."""
    from collections import Counter, defaultdict

    from adas_vla.config import load_config
    from adas_vla.deploy.qwen_vl_sim import DeployVLM
    from adas_vla.training.data import sample_visual
    from adas_vla.training.evaluate import select_records

    qv.LLM_MASK = args.mask
    cfg = load_config()
    out = Path(args.dir) / "calib"
    exported = json.loads((Path(args.dir) / "export.json").read_text())  # the graphs' rotations
    vlm = DeployVLM(MODEL, device=args.device, dtype=getattr(torch, args.dtype), rotate=exported["rotate"],
                    rotate_heads=exported.get("rotate_heads", False), gptq=exported.get("gptq"))
    spec, names = vlm.spec, [p.io_names()[0] for p in vlm.parts]
    lists, stats, steps = defaultdict(list), {}, Counter()

    def rng(key: str, t: torch.Tensor) -> None:
        lo, hi = stats.get(key, (float("inf"), float("-inf")))
        stats[key] = (min(lo, float(t.min())), max(hi, float(t.max())))

    def save(name: str, tensors: dict[str, torch.Tensor | np.ndarray]) -> None:
        d = out / name / f"{len(lists[name]):03d}"
        d.mkdir(parents=True, exist_ok=True)
        line = []
        for nm, t in tensors.items():
            arr = t.float().cpu().numpy() if torch.is_tensor(t) else t
            if isinstance(arr, Path):  # shared constant input
                line.append(f"{nm}:={arr}")
                continue
            arr.astype(np.float32).tofile(d / f"{nm}.raw")
            line.append(f"{nm}:={d / f'{nm}.raw'}")
        lists[name].append(" ".join(line))

    def on_part(p: int, ar: int, inputs, outputs, n: int) -> None:
        part = vlm.parts[p]
        if ar == 1:
            steps[p] += 1
            if (steps[p] - 1) % args.ar1_every or steps[p] > args.ar1_every * args.ar1_steps:
                return
        save(graph_name(p, ar, args.cl), dict(zip(names[p], inputs)))
        if p == 0:
            rng("inputs_embeds", inputs[0][0, :n])
        if not part.last:
            rng(part.io_names()[1][0], outputs[0][0, :n])
        for j, layer in enumerate(part.layer_ids):
            rng(f"past_key_{layer}", outputs[1 + 2 * j][..., :n])
            rng(f"past_value_{layer}", outputs[2 + 2 * j][:, :, :n])

    vlm.gen.on_part = on_part
    t0 = time.time()
    records = select_records(str(DATA), split="train")[::args.stride][:args.prompts]
    for k, rec in enumerate(records):
        steps.clear()
        image = sample_visual(rec, cfg)
        _, raw, _ = vlm.decide(image, rec["context"], rec["ego_speed_kmh"], rec["cruise_speed_kmh"])
        print(f"{k + 1}/{len(records)} {rec['id']} {time.time() - t0:.0f} s {raw}", flush=True)
    for name, lines in lists.items():
        (out / name / "input_list.txt").write_text("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} samples")
    stats["attention_mask"] = (args.mask, 0.0)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))


def cmd_calib_vit(args) -> None:
    """ViT calibration list (patches of train frames + the constant cos/sin/mask inputs) and a held-out list of val
    frames; needs no model, only the preprocessing."""
    from adas_vla.config import load_config
    from adas_vla.training.data import sample_visual
    from adas_vla.training.evaluate import select_records

    cfg, spec = load_config(), qv.Spec.from_model(MODEL)
    out = Path(args.dir) / "calib"
    consts = {}
    for k, v in qv.vit_inputs(spec).items():
        consts[k] = out / "vision_encoder" / f"{k}.raw"
        consts[k].parent.mkdir(parents=True, exist_ok=True)
        v.numpy().tofile(consts[k])
    sets = {"vision_encoder": select_records(str(DATA), split="train")[::args.stride][:args.frames],
            "vision_encoder_eval": val_records(args.eval_frames, 900 // max(1, args.eval_frames))}
    for name, records in sets.items():
        lines = []
        for k, rec in enumerate(records):
            d = out / name / f"{k:03d}"
            d.mkdir(parents=True, exist_ok=True)
            qv.patchify(qv.resize_rgb(sample_visual(rec, cfg), spec), spec).tofile(d / "pixel_values.raw")
            lines.append(" ".join([f"pixel_values:={d / 'pixel_values.raw'}"] +
                                  [f"{n}:={consts[n]}" for n in qv.VisionEncoder.INPUTS[1:]]))
        (out / name / "input_list.txt").write_text("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} frames")


def act_encoding(lo: float, hi: float, bits: int = 16) -> list[dict]:
    """Asymmetric unsigned activation encoding (the converter derives scale / offset from min, max)."""
    return [{"bitwidth": bits, "dtype": "int", "is_symmetric": "False", "min": float(lo), "max": float(hi)}]


def kv_encoding(absmax: float) -> list[dict]:
    """8-bit KV cache as in the AI Hub bundles: uint8 with offset -128 (symmetric around 0)."""
    s = max(absmax, 1e-6) / 127
    return [{"bitwidth": 8, "dtype": "int", "is_symmetric": "False", "min": -128 * s, "max": 127 * s,
             "scale": s, "offset": -128}]


def cmd_overrides(args) -> None:
    """Encoding overrides for the LLM graphs (qairt-converter --quantization_overrides): every tensor that crosses a
    graph boundary gets one encoding for all graphs that see it (KV cache of each layer, hidden state between parts,
    inputs_embeds, attention mask [mask, 0], RoPE tables [-1, 1]); the LM head weights 8-bit per channel."""
    stats = json.loads((Path(args.dir) / "calib/stats.json").read_text())
    spec = qv.Spec.from_model(MODEL)
    rot = qv.hadamard(spec.dim)
    qw = None
    if args.gptq:  # the projection weights already sit on their GPTQ grid: give QAIRT exactly that grid
        from adas_vla.deploy.gptq import QuantizedWeights

        qw = QuantizedWeights(args.gptq)
        if qw.meta.get("block"):
            raise SystemExit("LPBQ overrides need the v1.0.0 encoding format: not written yet")
    w4 = lambda sc: [{"bitwidth": 4, "dtype": "int", "is_symmetric": "True", "min": -8 * float(v),  # noqa: E731
                      "max": 7 * float(v), "scale": float(v), "offset": -8} for v in sc]
    margin = lambda lo, hi: (lo - args.margin * (hi - lo), hi + args.margin * (hi - lo))  # noqa: E731
    out = Path(args.dir) / "overrides"
    out.mkdir(parents=True, exist_ok=True)
    head = None
    ranges = qv.part_ranges(spec)
    for i, r in enumerate(ranges):
        last = i == len(ranges) - 1
        acts = {"attention_mask": act_encoding(*stats["attention_mask"]), "position_ids_cos": act_encoding(-1, 1),
                "position_ids_sin": act_encoding(-1, 1)}
        first = "inputs_embeds" if r[0] == 0 else f"hidden_{r[0]}"
        acts[first] = act_encoding(*margin(*stats[first]))
        if not last:
            acts[f"hidden_{r[-1] + 1}"] = act_encoding(*margin(*stats[f"hidden_{r[-1] + 1}"]))
        for layer in r:
            for kind in ("key", "value"):
                enc = kv_encoding(max(abs(v) for v in stats[f"past_{kind}_{layer}"]) * (1 + args.margin))
                acts[f"past_{kind}_{layer}_in"] = acts[f"past_{kind}_{layer}_out"] = enc
        params = {}
        for j, layer in enumerate(r) if qw is not None else ():
            hd = spec.head_dim
            for name in ("q", "k", "v", "o", "gate", "up", "down"):
                sc = qw.scales(layer, name)[:, 0]
                if name in ("q", "k", "v"):  # split heads (SHA): one conv per head
                    for h in range(sc.shape[0] // hd):
                        params[f"layers.{j}.{name}.{h}.weight"] = w4(sc[h * hd:(h + 1) * hd])
                else:
                    params[f"layers.{j}.{name}.weight"] = w4(sc)
        if last:
            if head is None:
                w = qv.head_weights(qv.Checkpoint(MODEL), rot)["lm_head"]
                s = w.abs().amax(1).clamp_min(1e-12) / 127
                head = [{"bitwidth": 8, "dtype": "int", "is_symmetric": "True", "min": -128 * float(v),
                         "max": 127 * float(v), "scale": float(v), "offset": -128} for v in s]
            params["lm_head.weight"] = head
        for ar in (args.ar, 1):
            name = graph_name(i, ar, args.cl)
            (out / f"{name}.json").write_text(json.dumps({"activation_encodings": acts, "param_encodings": params}))
            print("wrote", out / f"{name}.json")


SOCS = {"sa8650": {"soc_model": 52, "dsp_arch": "v73"}, "sa8797": {"soc_model": 72, "dsp_arch": "v81"}}
QAIRT = Path.home() / "workspaces/qairt_env/qairt.sh"


def qairt(cmd: list[str], cwd: Path, log: Path, mem: str = "16g") -> None:
    """Run a QAIRT host tool in the docker wrapper with the SSD folder mounted."""
    import os
    import subprocess

    env = dict(os.environ, WORK=str(QNN_DIR.parent), DOCKER_MEM=mem)
    with open(log, "w") as f:
        r = subprocess.run([str(QAIRT), *cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"{cmd[0]} failed, see {log}")


def sqnr_db(ref: np.ndarray, x: np.ndarray) -> float:
    return float(10 * np.log10((ref.astype(np.float64) ** 2).sum() / max(((ref - x).astype(np.float64) ** 2).sum(), 1e-30)))


def cmd_check_bins(args) -> None:
    """Encodings that Genie copies between graphs without requantizing must agree in the context binaries: the hidden
    state out of part i and into part i+1, the K / V cache of each layer in and out of the AR128 and AR1 graphs."""
    q = Path(args.dir) / "bin" / args.soc
    tensors: dict[str, set] = {}
    for f in sorted(q.glob("part*_of_4.json")):
        for g in json.loads(f.read_text())["info"]["graphs"]:
            gi = g["info"]
            for t in gi["graphInputs"] + gi["graphOutputs"]:
                ti = t["info"]
                so = ti.get("quantizeParams", {}).get("scaleOffset", {})
                key = ti["name"].replace("_in", "").replace("_out", "") if ti["name"].startswith("past_") else ti["name"]
                tensors.setdefault(key, set()).add((ti["dataType"], round(so.get("scale", 0), 10), so.get("offset")))
    bad = {k: v for k, v in tensors.items() if len(v) > 1 and (k.startswith(("past_", "hidden_")))}
    shared = [k for k in tensors if k.startswith(("past_", "hidden_"))]
    print(f"{args.soc}: {len(shared)} shared tensors (KV per layer, hidden states), {len(bad)} with different encodings")
    for k, v in sorted(bad.items()):
        print("  ", k, sorted(v))
    for k in ("inputs_embeds", "attention_mask", "position_ids_cos", "logits"):
        print("  ", k, sorted(tensors.get(k, [])))


def cmd_eval_part(args) -> None:
    """One quantized LLM graph on the x86 HTP emulator vs its float ONNX (onnxruntime) on calibration inputs:
    SQNR of every output (hidden state / logits, new K / V), top-1 agreement of the logits."""
    import onnxruntime as ort

    q = Path(args.dir)
    d = q / args.name
    out = d / f"emu_{args.soc}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "htp_ext.json").write_text(json.dumps({"devices": [{"soc_id": SOCS[args.soc]["soc_model"],
                                                              "dsp_arch": SOCS[args.soc]["dsp_arch"]}]}))
    (out / "htp_config.json").write_text(json.dumps({"backend_extensions": {
        "shared_library_path": "libQnnHtpNetRunExtensions.so", "config_file_path": str(out / "htp_ext.json")}}))
    lines = (q / "calib" / args.name / "input_list.txt").read_text().strip().splitlines()
    lines = lines[args.first:args.first + args.samples]
    (out / "input_list.txt").write_text("\n".join(lines) + "\n")
    t0 = time.time()
    qairt(["qnn-net-run", "--backend", "libQnnHtp.so", "--model", "libQnnModelDlc.so", "--dlc_path",
           str(d / f"{args.name}.q.dlc"), "--input_list", str(out / "input_list.txt"), "--output_dir", str(out),
           "--config_file", str(out / "htp_config.json")], d, out / "net_run.log")
    emu_s = time.time() - t0
    sess = ort.InferenceSession(str(d / f"{args.name}.onnx"), providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    names = [o.name for o in sess.get_outputs()]
    rows = {}
    top1 = []
    for k, line in enumerate(lines):
        feeds = {n: np.fromfile(p, np.float32).reshape(shapes[n]) for n, p in (x.split(":=") for x in line.split())}
        ref = sess.run(None, feeds)
        for n, r in zip(names, ref):
            y = np.fromfile(out / f"Result_{k}" / f"{n}.raw", np.float32).reshape(r.shape)
            rows.setdefault(n, []).append(sqnr_db(r, y))
            if n == "logits":
                top1.append(float((r[0].argmax(-1) == y[0].argmax(-1)).mean()))
    kv = [np.mean(v) for n, v in rows.items() if n.startswith("past_")]
    main = names[0]
    print(f"{args.name} on the {args.soc} emulator: {len(lines)} samples, {emu_s:.0f} s; SQNR {main} "
          f"{np.mean(rows[main]):.1f} dB (min {np.min(rows[main]):.1f}), KV outputs mean {np.mean(kv):.1f} dB "
          f"(worst {min(kv):.1f})" + (f", logits top-1 agreement {np.mean(top1):.1%}" if top1 else ""))


def cmd_eval_vit(args) -> None:
    """Quantized ViT on the x86 HTP emulator (qnn-net-run) vs the float ONNX outputs of the val frames: SQNR and
    per-token cosine, as for the InternVL ViT (board/README.md)."""
    q = Path(args.dir)
    out = q / "vision_encoder" / f"emu_{args.soc}"
    out.mkdir(parents=True, exist_ok=True)
    ext = out / "htp_ext.json"
    ext.write_text(json.dumps({"devices": [{"soc_id": SOCS[args.soc]["soc_model"],
                                           "dsp_arch": SOCS[args.soc]["dsp_arch"]}]}))
    cfg = out / "htp_config.json"
    cfg.write_text(json.dumps({"backend_extensions": {"shared_library_path": "libQnnHtpNetRunExtensions.so",
                                                      "config_file_path": str(ext)}}))
    lst = q / "calib/vision_encoder_eval/input_list.txt"
    if args.frames:  # the emulator takes minutes per frame
        short = out / "input_list.txt"
        short.write_text("\n".join(lst.read_text().strip().splitlines()[:args.frames]) + "\n")
        lst = short
    t0 = time.time()
    qairt(["qnn-net-run", "--backend", "libQnnHtp.so", "--model", "libQnnModelDlc.so", "--dlc_path",
           str(q / "vision_encoder" / args.dlc), "--input_list", str(lst), "--output_dir", str(out),
           "--config_file", str(cfg)], q / "vision_encoder", out / "net_run.log")
    rows = []
    for i, line in enumerate(lst.read_text().strip().splitlines()):
        ref = np.fromfile(Path(line.split()[0].split(":=")[1]).parent / "image_features.float.raw", np.float32)
        y = np.fromfile(out / f"Result_{i}" / "image_features.raw", np.float32)
        ref, y = ref.reshape(-1, 2048), y.reshape(-1, 2048)
        cos = (ref * y).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
        rows.append((sqnr_db(ref, y), float(cos.mean()), float(cos.min())))
    r = np.array(rows)
    print(f"{args.dlc} on the {args.soc} emulator ({time.time() - t0:.0f} s, {len(rows)} frames): SQNR "
          f"{r[:, 0].mean():.1f} dB (min {r[:, 0].min():.1f}), token cosine mean {r[:, 1].mean():.4f}, "
          f"worst token {r[:, 2].min():.4f}")



def cmd_bundle(args) -> None:
    """Genie bundle for the board from the build outputs (bin/<soc>/*.bin): context binaries, LUT, tokenizer, the
    constant ViT inputs and the Genie configs, modelled on the AI Hub qwen2_5_vl_7b_instruct bundle (same node
    types and M-RoPE settings): ViT on NSP0, LLM on NSP1 (board/vlm_server.sh, MODEL=v3)."""
    import shutil

    spec = qv.Spec.from_model(MODEL)
    q = Path(args.dir)
    out = q / "bundle" / args.soc
    out.mkdir(parents=True, exist_ok=True)
    bins = ["vision_encoder.bin"] + [f"part{i}_of_4.bin" for i in range(1, 5)]
    for b in bins:
        src = q / "bin" / args.soc / b
        if not src.exists():
            raise SystemExit(f"missing {src}: run scripts/v3_qnn_build.sh ctx {args.soc}")
        (out / b).unlink(missing_ok=True)
        (out / b).symlink_to(src)
    for name in ("embedding_weights.raw",):
        (out / name).unlink(missing_ok=True)
        (out / name).symlink_to(q / name)
    shutil.copy(MODEL / "tokenizer.json", out / "tokenizer.json")
    for k, v in qv.vit_inputs(spec).items():
        v.numpy().tofile(out / f"vit_{k}.raw")
    dev = lambda i: {"devices": [{**SOCS[args.soc], "device_id": i, "cores": [  # noqa: E731
        {"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}]}],
        "memory": {"mem_type": "shared_buffer"}, "context": {"weight_sharing_enabled": True}}
    htp = lambda ext, mmap: {"version": 1, "type": "QnnHtp", "QnnHtp": {  # noqa: E731
        "version": 1, "use-mmap": mmap, "spill-fill-bufsize": 0, "mmap-budget": 0, "poll": False,
        "cpu-mask": "0xe0", "kv-dim": spec.head_dim, "allow-async-init": False, "enable-graph-switching": False},
        "extensions": ext}
    image = {"image-encoder": {"version": 1, "engine": {"version": 1, "mode": "image", "backend": {
        "version": 1, "type": "QnnHtp", "QnnHtp": {"version": 1, "spill-fill-bufsize": 0, "use-mmap": False,
                                                   "allow-async-init": False},
        "extensions": "htp_ext_nsp0.json"}, "model": {"version": 1, "type": "binary", "binary": {
            "version": 1, "ctx-bins": ["vision_encoder.bin"]},
            "vision-param": {"height": spec.grid_h, "width": spec.grid_w}}}}}
    lut = {"version": 1, "lut-path": "embedding_weights.raw", "size": spec.dim, "datatype": "float32"}
    text = {"text-encoder": {"version": 1, "type": "lut", "lut": lut,
                             "tokenizer": {"version": 1, "path": "tokenizer.json"}}}
    model = {"version": 1, "type": "binary", "binary": {"version": 1, "ctx-bins": bins[1:]},
             "positional-encoding": {"type": "rope", "rope-dim": spec.head_dim // 2, "rope-theta": int(spec.theta),
                                     "rope-scaling": {"rope-type": "qwen2vl-mrope", "time-step": 50,
                                                      "spatial-merge-size": spec.merge,
                                                      "mrope-section": list(spec.mrope)}}}
    gen = {"text-generator": {
        "version": 1, "type": "basic",
        "context": {"version": 1, "size": args.cl, "n-vocab": spec.vocab, "bos-token": 151643,
                    "eos-token": list(spec.eos)},  # <|im_end|> and <|endoftext|>, as generation_config.json
        "sampler": {"version": 1, "seed": 42, "temp": 0.0001, "top-k": 1, "top-p": 0.95},
        "tokenizer": {"version": 1, "path": "tokenizer.json"},
        "engine": {"version": 1, "n-threads": 3, "backend": htp("htp_ext_nsp1.json", True), "model": model},
        "embedding": {"version": 1, "type": "lut", "lut-path": "embedding_weights.raw", "size": spec.dim,
                      "datatype": "float32"},
        "max-num-tokens": 64}}
    fast = json.loads(json.dumps(gen))
    fast["text-generator"]["stop-sequence"] = ['"reason"']
    dialog = {"dialog": {k: v for k, v in json.loads(json.dumps(gen))["text-generator"].items()
                         if k not in ("max-num-tokens",)}}
    dialog["dialog"]["engine"]["backend"]["extensions"] = "htp_ext_nsp0.json"
    configs = {"htp_ext_nsp0.json": dev(0), "htp_ext_nsp1.json": dev(1), "img-enc-nsp0.json": image,
               "text-encoder.json": text, "greedy-text-generator-nsp1.json": gen,
               "fast-text-generator-nsp1.json": fast, "genie_config.json": dialog}
    for name, cfg in configs.items():
        (out / name).write_text(json.dumps(cfg, indent=2))
    size = sum((out / b).stat().st_size for b in bins) + (q / "embedding_weights.raw").stat().st_size
    print(f"bundle {out}: {len(configs)} configs, {size / 2**30:.2f} GiB of binaries + LUT")


def cmd_gptq(args) -> None:
    """GPTQ 4-bit weights of the rotated LLM (deploy/gptq.py) from train decision prompts with their answers, one
    layer at a time on the GPU -> <dir>/gptq_w4[_lpbq64].safetensors (int8 codes + scales)."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from transformers import AutoProcessor

    from adas_vla.config import load_config
    from adas_vla.deploy import gptq as G
    from adas_vla.deploy.qwen_vl_sim import split_prompt
    from adas_vla.training.data import sample_visual, target_json
    from adas_vla.training.evaluate import select_records

    qv.LLM_MASK = -1e4
    cfg, spec, ck = load_config(), qv.Spec.from_model(MODEL), qv.Checkpoint(MODEL)
    exported = json.loads((Path(args.dir) / "export.json").read_text())
    rot = qv.hadamard(spec.dim) if exported["rotate"] else None
    hr = qv.hadamard(spec.head_dim, seed=1) if exported.get("rotate_heads") else None
    dev = torch.device("cuda")
    processor, tok = AutoProcessor.from_pretrained(MODEL), Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    lut = qv.embedding_table(ck, rot)
    vit = qv.VisionEncoder(ck, spec, rot, torch.bfloat16).to(dev).eval()
    vit_in = [t.to(dev, torch.bfloat16) for t in qv.vit_inputs(spec).values()]
    prompts = []
    t0 = time.time()
    for rec in select_records(str(DATA), split="train")[args.offset::args.stride][:args.prompts]:
        image = sample_visual(rec, cfg)
        text = processor.apply_chat_template(messages_of(rec, image), tokenize=False, add_generation_prompt=True)
        before, after = split_prompt(text)
        answer = target_json(rec["target"])  # the decoded part too: decision prefix + a typical answer
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
        ids_b, ids_a = enc(before), enc(after + answer)
        with torch.no_grad():
            pix = torch.from_numpy(qv.patchify(qv.resize_rgb(image, spec), spec)).to(dev, torch.bfloat16)
            feats = vit(pix, *vit_in).float().cpu()
        x = torch.cat([lut[ids_b], feats, lut[ids_a]])[None].to(dev)
        prompts.append((x, *G.causal_inputs(spec, qv.prompt_positions(len(ids_b), len(ids_a), spec), dev)))
    del vit
    torch.cuda.empty_cache()
    print(f"{len(prompts)} calibration prompts, {sum(p[0].shape[1] for p in prompts)} tokens, "
          f"{time.time() - t0:.0f} s", flush=True)
    tensors, report = {}, {}
    for layer_id in range(spec.layers):
        t1 = time.time()
        layer = qv.DecoderLayer(qv.layer_weights(ck, layer_id, rot, torch.float32, hr), spec, hr).to(dev).eval()
        outs, grids, errors = G.quantize_layer(layer, prompts, args.bits, args.block)
        G.save_layer(tensors, layer_id, G.convs_of(layer), grids, args.bits, args.block)
        prompts = [(o, *p[1:]) for o, p in zip(outs, prompts)]
        report[layer_id] = errors
        worst = max(errors, key=lambda k: errors[k][0])
        print(f"layer {layer_id:2d} {time.time() - t1:5.1f} s  output error GPTQ / RTN: "
              + "  ".join(f"{k} {g:.4f}/{r:.4f}" for k, (g, r) in errors.items()) + f"  (worst {worst})", flush=True)
        del layer
        torch.cuda.empty_cache()
    tag = f"gptq_w{args.bits}" + (f"_lpbq{args.block}" if args.block else "")
    meta = {"bits": args.bits, "block": args.block, "prompts": len(prompts), **exported}
    save_file(tensors, str(Path(args.dir) / f"{tag}.safetensors"), metadata={"gptq": json.dumps(meta)})
    (Path(args.dir) / f"{tag}.report.json").write_text(json.dumps(report, indent=1))
    print("wrote", Path(args.dir) / f"{tag}.safetensors")


def cmd_eval_board(args) -> None:
    """The val split through v3 on the board (board/vlm_genie_server.cpp, MODEL=v3 board/vlm_server.sh start):
    the same metrics and report as `eval`, plus the server's timings per sample (<report>.timing.jsonl)."""
    from adas_vla.config import load_config
    from adas_vla.reasoning.board_vlm import BoardVLM
    from adas_vla.training.evaluate import evaluate

    cfg = load_config()
    cfg.vlm.backend, cfg.vlm.board_processor = "board", str(MODEL)
    cfg.vlm.board_host, cfg.vlm.board_port = args.host, args.port
    timings = []

    class TimedBoardVLM(BoardVLM):
        def decide(self, *a, **k):
            out = super().decide(*a, **k)
            timings.append({**self.last_timing, "round_trip_ms": out[2] * 1000})
            return out

    vlm = TimedBoardVLM(cfg.vlm)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(cfg, str(DATA), limit=args.limit, report_path=args.report, split="val", vlm=vlm,
                       offset=args.offset)
    Path(args.report).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=1))
    with open(Path(args.report).with_suffix(".timing.jsonl"), "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in timings)
    keys = ("image_ms", "first_token_ms", "total_ms", "tokens", "round_trip_ms")
    print("board timing p50:", {k: float(np.median([t[k] for t in timings])) for k in keys})


def cmd_compare(args) -> None:
    """Two or more eval JSONL reports on their common samples: agreement with the first (the reference) and the
    project's metrics (joint accuracy, under-braking first) of each."""
    from adas_vla.training.evaluate import under_brakes
    from adas_vla.types import LongAction

    runs = [{json.loads(line)["image"]: json.loads(line) for line in open(p)} for p in args.reports]
    common = sorted(set.intersection(*(set(r) for r in runs)))
    rank = lambda a: LongAction(a)  # noqa: E731
    print(f"{len(common)} common samples")
    ref = runs[0]
    for path, run in zip(args.reports, runs):
        n = len(common)
        valid = [k for k in common if run[k]["pred"]]
        joint = sum(run[k]["pred"]["longitudinal"] == run[k]["gt"]["longitudinal"]
                    and run[k]["pred"]["lateral"] == run[k]["gt"]["lateral"] for k in valid)
        under = sum(under_brakes(rank(run[k]["gt"]["longitudinal"]), rank(run[k]["pred"]["longitudinal"]))
                    for k in valid)
        same_long = sum(bool(run[k]["pred"]) and bool(ref[k]["pred"])
                        and run[k]["pred"]["longitudinal"] == ref[k]["pred"]["longitudinal"] for k in common)
        same_joint = sum(bool(run[k]["pred"]) and bool(ref[k]["pred"])
                         and run[k]["pred"]["longitudinal"] == ref[k]["pred"]["longitudinal"]
                         and run[k]["pred"]["lateral"] == ref[k]["pred"]["lateral"] for k in common)
        speed = [abs(run[k]["pred"]["target_speed_kmh"] - ref[k]["pred"]["target_speed_kmh"]) for k in common
                 if run[k]["pred"] and ref[k]["pred"]]
        print(f"{Path(path).name:38s} valid {len(valid) / n:6.1%}  joint {joint / n:6.1%}  under-braking "
              f"{under / n:5.1%}  | vs ref: longitudinal {same_long / n:6.1%}  joint {same_joint / n:6.1%}  "
              f"speed |d| {np.mean(speed) if speed else 0:4.1f} km/h")


def cmd_vit_cache(args) -> None:
    """float32 ViT features of every val image on the GPU (the ViT alone fits in float32), for `eval --vit-cache`:
    the float LLM with a float32 ViT instead of the bf16 one the PC pipeline uses (15 dB from float32)."""
    from adas_vla.config import load_config
    from adas_vla.deploy.qwen_vl_sim import DeployVLM
    from adas_vla.training.data import sample_visual
    from adas_vla.training.evaluate import select_records

    cfg, spec, ck = load_config(), qv.Spec.from_model(MODEL), qv.Checkpoint(MODEL)
    rot = qv.hadamard(spec.dim) if args.rotate else None
    dev = torch.device("cuda")
    vit = qv.VisionEncoder(ck, spec, rot, torch.float32).to(dev).eval()
    vit_in = [t.to(dev) for t in qv.vit_inputs(spec).values()]
    cache = {}
    with torch.no_grad():
        for rec in select_records(str(DATA), split="val"):
            rgb = qv.resize_rgb(sample_visual(rec, cfg), spec)
            pix = torch.from_numpy(qv.patchify(rgb, spec)).to(dev)
            cache[DeployVLM.image_key(rgb)] = vit(pix, *vit_in).cpu()
    torch.save(cache, args.out)
    print(f"{len(cache)} images -> {args.out}")


def cmd_eval(args) -> None:
    """The val split through the deployment form on the GPU (float, or simulated quantization), scored by
    training.evaluate like `adas-vla eval` (same metrics, same JSONL report)."""
    from adas_vla.config import load_config
    from adas_vla.deploy.qwen_vl_sim import DeployVLM
    from adas_vla.training.data import sample_visual
    from adas_vla.training.evaluate import evaluate, select_records

    cfg = load_config()
    t0 = time.time()
    vlm = DeployVLM(MODEL, rotate=args.rotate, wbits=args.wbits, wblock=args.wblock, kv_bits=args.kv_bits,
                    rotate_heads=args.rotate_heads, gptq=args.gptq,
                    feature_cache=torch.load(args.vit_cache) if args.vit_cache else None)
    print(f"deployment form on {vlm.device}: {time.time() - t0:.0f} s, "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if vlm.kv is not None:  # KV scales from train samples, as the QNN encodings will be
        for rec in select_records(str(DATA), split="train")[::400][:args.calib]:
            vlm.decide(sample_visual(rec, cfg), rec["context"], rec["ego_speed_kmh"], rec["cruise_speed_kmh"])
        vlm.kv.calibrating = False
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(cfg, str(DATA), limit=args.limit, report_path=args.report, split="val", vlm=vlm,
                       offset=args.offset)
    Path(args.report).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("what", nargs="+", choices=["vit", "part1", "part2", "part3", "part4", "lut"])
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--rotate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rotate-heads", action="store_true", help="also R2 (V/O) + R3 (Q/K after RoPE), 128x128")
    p.add_argument("--rotate-vit", action=argparse.BooleanOptionalAction, default=True,
                   help="rotate the ViT residual stream (its outliers reach 2.8e4)")
    p.add_argument("--gptq", help="GPTQ weights file (scripts/v3_qnn.py gptq) for the LLM projections")
    p.add_argument("--cl", type=int, default=1024)
    p.add_argument("--ar", type=int, default=128)
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("calib")
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--prompts", type=int, default=12, help="train prompts (AR128: ~5 chunks each)")
    p.add_argument("--stride", type=int, default=1000, help="take every stride-th train record (mixes the sources)")
    p.add_argument("--ar1-steps", type=int, default=4, help="decode steps saved per prompt for the AR1 graphs")
    p.add_argument("--ar1-every", type=int, default=4, help="save every n-th decode step")
    p.add_argument("--mask", type=float, default=-1000.0, help="additive attention-mask value in the LLM graphs")
    p.add_argument("--cl", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.set_defaults(func=cmd_calib)
    p = sub.add_parser("calib-vit")
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--stride", type=int, default=250)
    p.add_argument("--eval-frames", type=int, default=12)
    p.set_defaults(func=cmd_calib_vit)
    p = sub.add_parser("overrides")
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--margin", type=float, default=0.05, help="relative margin on calibrated ranges")
    p.add_argument("--gptq", help="GPTQ weights file: 4-bit per-channel encodings of every projection")
    p.add_argument("--cl", type=int, default=1024)
    p.add_argument("--ar", type=int, default=128)
    p.set_defaults(func=cmd_overrides)
    p = sub.add_parser("check-bins")
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--soc", default="sa8650", choices=["sa8650", "sa8797"])
    p.set_defaults(func=cmd_check_bins)
    p = sub.add_parser("eval-part")
    p.add_argument("name", help="graph name, e.g. ar1_cl1024_1_of_4")
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--first", type=int, default=0, help="first calibration sample")
    p.add_argument("--soc", default="sa8650", choices=["sa8650", "sa8797"])
    p.set_defaults(func=cmd_eval_part)
    p = sub.add_parser("eval-vit")
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--dlc", default="vision_encoder.q.dlc")
    p.add_argument("--frames", type=int, help="only the first N eval frames")
    p.add_argument("--soc", default="sa8650", choices=["sa8650", "sa8797"])
    p.set_defaults(func=cmd_eval_vit)
    p = sub.add_parser("gptq")
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--block", type=int, help="LPBQ block size (default: per output channel)")
    p.add_argument("--prompts", type=int, default=48)
    p.add_argument("--stride", type=int, default=250)
    p.add_argument("--offset", type=int, default=7, help="first train record (away from the calib prompts)")
    p.set_defaults(func=cmd_gptq)
    p = sub.add_parser("eval-board")
    p.add_argument("--report", default=str(ROOT / "outputs/v3_qnn/eval_board_sa8650.jsonl"))
    p.add_argument("--limit", type=int)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--host", default="192.168.0.73")
    p.add_argument("--port", type=int, default=50053)
    p.set_defaults(func=cmd_eval_board)
    p = sub.add_parser("compare")
    p.add_argument("reports", nargs="+", help="eval JSONL reports, the first is the reference")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("bundle")
    p.add_argument("soc", choices=["sa8650", "sa8797"])
    p.add_argument("--dir", default=str(QNN_DIR))
    p.add_argument("--cl", type=int, default=1024)
    p.set_defaults(func=cmd_bundle)
    p = sub.add_parser("vit-cache")
    p.add_argument("--out", default=str(QNN_DIR / "vit_fp32_val.pt"))
    p.add_argument("--rotate", action="store_true")
    p.set_defaults(func=cmd_vit_cache)
    p = sub.add_parser("eval")
    p.add_argument("--report", default=str(ROOT / "outputs/v3_qnn/eval_float.jsonl"))
    p.add_argument("--limit", type=int)
    p.add_argument("--offset", type=int, default=0, help="skip the first val records (resume a run)")
    p.add_argument("--rotate", action="store_true", help="Hadamard rotation of the LLM residual (R1)")
    p.add_argument("--rotate-heads", action="store_true", help="also R2 (V/O) + R3 (Q/K after RoPE)")
    p.add_argument("--wbits", type=int, help="simulate LLM weights at this precision (LM head 8-bit)")
    p.add_argument("--wblock", type=int, help="LPBQ block size (default: per output channel)")
    p.add_argument("--kv-bits", type=int, help="simulate the KV cache at this precision")
    p.add_argument("--gptq", help="GPTQ weights file (scripts/v3_qnn.py gptq), needs --rotate like the export")
    p.add_argument("--vit-cache", help="image features from vit-cache (same --rotate) instead of the bf16 ViT")
    p.add_argument("--calib", type=int, default=16, help="train samples to calibrate the KV scales")
    p.set_defaults(func=cmd_eval)
    for name, fn in (("check-hf", cmd_check_hf), ("check-deploy", cmd_check_deploy)):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=3, help="val samples")
        p.add_argument("--stride", type=int, default=300, help="take every stride-th val sample")
        p.add_argument("--tokens", type=int, default=24, help="greedy tokens to compare")
        p.add_argument("--threads", type=int, default=12)
        p.add_argument("--mask", type=float, help="additive LLM mask value (check-deploy; default qwen_vl.LLM_MASK)")
        p.add_argument("--out", default=str(ROOT / "outputs/v3_qnn"))
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
