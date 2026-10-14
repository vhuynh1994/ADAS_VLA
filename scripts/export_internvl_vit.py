#!/usr/bin/env python3
"""Export the InternVL3.5-2B vision encoder (InternViT-300M + pixel shuffle + MLP projector) for our own QAIRT build.

The Qualcomm AI Hub Genie bundle ships the vision encoder already compiled (`vision_encoder.bin`, W8A16, O3, VTCM 8,
DLBC off). Rebuilding it ourselves lets us change compile options (DLBC, ...) and quantization. The graph is the one
of the AI Hub recipe (qai_hub_models templates/intern3_5_vl/vision_encoder.py): the remote-code InternViT with its
Linear layers as 1x1 Conv2d (replace_linears_with_convs), the last hidden state without the CLS token, pixel shuffle
0.5 and mlp1; input `pixel_values` [1,3,448,448] float32 (ImageNet-normalised RGB), output `image_features`
[1,256,2048] (Genie only treats a graph as an image encoder when an output is named image_features*).

The bundle's LLM works in a rotated residual space: its embedding table is embed_tokens @ R with R an orthogonal
2048x2048 Hadamard matrix (+-1/sqrt(2048), QuaRot/SpinQuant-style rotation that spreads outliers before the W4
quantization), and its vision encoder outputs features @ R. R is recovered from the bundle (`--bundle`: orthogonal
Procrustes between the original embed_tokens and embedding_weights.raw, exact to ~1e-8) and folded into the last
projector Linear (W' = R^T W, b' = b R), so the rebuilt encoder can replace vision_encoder.bin at no extra cost.

Writes <out>/internvl_vit.onnx (+ external weights), calibration and evaluation inputs (float32 raw, same
preprocessing as adas_vla/reasoning/board_vlm.py) with QAIRT input lists, and the float reference outputs of the
evaluation frames. Needs qai_hub_models, so run it with the AI Hub venv:

  ~/workspaces/aihub/.venv/bin/python scripts/export_internvl_vit.py --out outputs/deploy_sa8650/internvl_vit
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# (video, start s, step s, frames): highway, city, UK, Australian roads, Nexar near-crash clips
CALIB_SOURCES = [
    ("data/samples/highway_traffic.mp4", 1.0, 3.0, 12),
    ("data/samples/highway_short.mp4", 0.5, 2.0, 6),
    ("data/raw/aap9002--UK-Road-DashCam/241220_125301_002_FH.MP4", 5.0, 9.0, 12),
    ("data/raw/qutegocentric--Australian_Roads_Dashcam_Driving/*/*.mp4", 2.0, 0.0, 18),  # first 18 by name
    ("data/raw/nexar-ai--nexar_collision_prediction/train/positive/000*.mp4", 10.0, 0.0, 16),
]
EVAL_SOURCES = [
    ("data/samples/highway_traffic.mp4", 2.5, 3.0, 6),
    ("data/raw/aap9002--UK-Road-DashCam/241220_130801_007_FH.MP4", 7.0, 11.0, 6),
    ("data/raw/nexar-ai--nexar_collision_prediction/train/positive/0100*.mp4", 5.0, 0.0, 8),
]


def frames(sources):
    """BGR frames: `step` > 0 -> several frames of one video, `step` == 0 -> one frame from each matching video."""
    import cv2

    for pattern, start, step, n in sources:
        paths = sorted(glob.glob(str(ROOT / pattern)))
        if not paths:
            print(f"skipped (not found): {pattern}", file=sys.stderr)
            continue
        jobs = [(paths[0], start + i * step) for i in range(n)] if step else [(p, start) for p in paths[:n]]
        for path, t in jobs:
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            cap.release()
            if ok:
                yield frame


def pixel_values(frame_bgr) -> np.ndarray:
    """Same preprocessing as board_vlm.pixel_values: bicubic 448x448, /255, ImageNet mean/std, NCHW float32."""
    from PIL import Image

    rgb = np.asarray(Image.fromarray(frame_bgr[:, :, ::-1]).resize((448, 448), Image.BICUBIC), np.uint8)
    x = rgb.astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def bundle_rotation(checkpoint: str, bundle: str) -> np.ndarray:
    """R with embedding_weights.raw = embed_tokens @ R (orthogonal Procrustes on 30k rows, checked on 5k others)."""
    from safetensors import safe_open

    lut = np.memmap(Path(bundle) / "embedding_weights.raw", np.float32, mode="r")
    with safe_open(str(Path(checkpoint) / "model.safetensors"), "pt") as f:
        emb = f.get_tensor("language_model.model.embed_tokens.weight").float().numpy()
    lut = lut.reshape(emb.shape)
    rng = np.random.default_rng(0)
    rows, test = rng.choice(len(emb), 30000, replace=False), rng.choice(len(emb), 5000, replace=False)
    u, _, vt = np.linalg.svd(emb[rows].astype(np.float64).T @ np.asarray(lut[rows], np.float64))
    rot = u @ vt
    err = np.linalg.norm(emb[test] @ rot - lut[test]) / np.linalg.norm(lut[test])
    if err > 1e-5:
        raise ValueError(f"the bundle's embeddings are not a rotation of the checkpoint's (relative error {err:.2e})")
    print(f"bundle rotation recovered: relative error {err:.1e}, |entries| {np.abs(rot).min():.6f}..{np.abs(rot).max():.6f}")
    return rot


def fold_rotation(projector: torch.nn.Module, rot: np.ndarray) -> None:
    """Make the projector output y @ R: last Linear y = x W^T + b -> W' = R^T W, b' = b R."""
    last = [m for m in projector.modules() if isinstance(m, torch.nn.Linear)][-1]
    r = torch.from_numpy(rot).double()
    with torch.no_grad():
        last.weight.copy_((r.T @ last.weight.double()).float())
        last.bias.copy_((last.bias.double() @ r).float())


def load_encoder(checkpoint: str, rot: np.ndarray | None = None):
    from transformers import AutoModel, PreTrainedModel

    from qai_hub_models.models.templates.intern3_5_vl.vision_encoder import Intern3_5VLVisionEncoder
    from qai_hub_models.models.templates.intern3_5_vl.vision_encoder_adaptations import (
        replace_visual_attention_with_adaptation,
    )
    from qai_hub_models.models.templates.qwen2_vl.vision_encoder_adaptations import replace_linears_with_convs

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):  # same workaround as the recipe
        type.__setattr__(PreTrainedModel, "all_tied_weights_keys", {})
    # bf16 load of the whole VLM (~4.3 GB of RAM instead of ~8.5 GB in fp32), keep only the vision part in fp32
    full = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16, use_flash_attn=False)
    visual, projector = full.vision_model.float(), full.mlp1.float()
    ratio, layer, ps = float(full.downsample_ratio), int(full.select_layer), str(full.ps_version)
    patch = full.config.vision_config.patch_size
    del full
    if rot is not None:
        fold_rotation(projector, rot)
    replace_visual_attention_with_adaptation(visual)
    replace_linears_with_convs(visual)
    replace_linears_with_convs(projector)
    grid = torch.tensor([[1, 448 // patch, 448 // patch]])
    enc = Intern3_5VLVisionEncoder(visual, projector, grid, patch_size=patch, downsample_ratio=ratio,
                                   select_layer=layer, ps_version=ps)
    return enc.eval()


class Exportable(torch.nn.Module):  # pixel_values -> image_features only
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, pixel_values):
        return self.enc(pixel_values)[0]


def write_inputs(out: Path, name: str, sources) -> list[np.ndarray]:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    xs = [pixel_values(f) for f in frames(sources)]
    for i, x in enumerate(xs):
        x.tofile(d / f"{i:03d}.raw")
    (out / f"{name}_list.txt").write_text("".join(f"{name}/{i:03d}.raw\n" for i in range(len(xs))))
    return xs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "models/OpenGVLab--InternVL3_5-2B"))
    ap.add_argument("--out", default=str(ROOT / "outputs/deploy_sa8650/internvl_vit"))
    ap.add_argument("--bundle", default=str(Path.home() / "workspaces/aihub/bundles/internvl/"
                                            "intern3_5_vl_2b-genie-w4a16-qualcomm_sa8775p"),
                    help="AI Hub Genie bundle whose LLM the encoder feeds (embedding rotation); '' = no rotation")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calib = write_inputs(out, "calib", CALIB_SOURCES)
    evals = write_inputs(out, "eval", EVAL_SOURCES)
    print(f"{len(calib)} calibration and {len(evals)} evaluation frames")

    rot = bundle_rotation(args.checkpoint, args.bundle) if args.bundle else None
    model = Exportable(load_encoder(args.checkpoint, rot))
    with torch.no_grad():
        ref = [model(torch.from_numpy(x)).numpy() for x in evals]
    (out / "ref").mkdir(exist_ok=True)
    for i, y in enumerate(ref):
        y.astype(np.float32).tofile(out / "ref" / f"{i:03d}.raw")
    print("float reference:", ref[0].shape)

    onnx_path = out / "internvl_vit.onnx"
    torch.onnx.export(model, (torch.from_numpy(evals[0]),), str(onnx_path), input_names=["pixel_values"],
                      output_names=["image_features"], opset_version=17, dynamo=False)
    import onnx

    m = onnx.load(str(onnx_path))  # weights > 2 GB protobuf limit is not an issue (1.2 GB), but keep them external
    onnx.save(m, str(onnx_path), save_as_external_data=True, location="internvl_vit.onnx.data")
    print("wrote", onnx_path)


if __name__ == "__main__":
    main()
