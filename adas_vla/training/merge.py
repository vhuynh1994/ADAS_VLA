"""Merge a LoRA adapter into the base weights shard by shard.

`adas-vla merge --full` instantiates the whole model in bf16 on the CPU and calls peft's merge_and_unload: 16 GB of
RAM for the 7B, more than a laptop has. The merged weight of a LoRA'd Linear is just `W + (alpha / r) * B @ A`, which
needs the two small factors and the one shard holding W, so this module streams the base safetensors shards one at
a time, adds the deltas and writes them back under the same file names and index. Everything else in the base
folder (config, tokenizer, processor, chat template) is copied. Checkpoint key renames that transformers applies at
load time (Qwen2.5-VL: file `model.layers.*` <-> module `model.language_model.layers.*`) are undone when matching.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
ADAPTER_FILE = "adapter_model.safetensors"
LORA_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<module>.+?)\.lora_(?P<ab>[AB])(?:\.[\w-]+)?\.weight$")
# Module path as peft saw it at training time -> how the same tensor may be named in the checkpoint files.
RENAMES = [("model.language_model.", "model."), ("model.visual.", "visual."), ("language_model.model.", "model.")]


def base_tensor_candidates(module: str) -> list[str]:
    key = module + ".weight"
    out = [key]
    for src, dst in RENAMES:
        if src in key:
            out.append(key.replace(src, dst, 1))
    return out


def lora_scale(adapter_cfg: dict, module: str, r: int) -> float:
    """peft's scaling: alpha / r (alpha / sqrt(r) with rsLoRA), with `alpha_pattern` overrides per module."""
    alpha = adapter_cfg.get("lora_alpha", r)
    for pattern, value in (adapter_cfg.get("alpha_pattern") or {}).items():
        if pattern == module or re.search(pattern, module):
            alpha = value
            break
    return alpha / math.sqrt(r) if adapter_cfg.get("use_rslora") else alpha / r


def read_adapter(adapter_dir: Path) -> tuple[dict, dict[str, dict[str, object]]]:
    """adapter_config.json and {module: {"A": tensor, "B": tensor}} of a LoRA adapter saved by peft."""
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("peft_type", "LORA") != "LORA":
        raise NotImplementedError(f"only LoRA adapters can be merged here, got {cfg.get('peft_type')}")
    if cfg.get("bias", "none") != "none" or cfg.get("modules_to_save"):
        raise NotImplementedError("adapters with trained biases or modules_to_save need `adas-vla merge --full`")
    path = adapter_dir / ADAPTER_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (only safetensors adapters are supported)")
    pairs: dict[str, dict[str, object]] = {}
    for key, tensor in load_file(str(path)).items():
        m = LORA_KEY.match(key)
        if m is None:
            raise ValueError(f"unexpected tensor in the adapter: {key}")
        pairs.setdefault(m["module"], {})[m["ab"]] = tensor
    for module, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise ValueError(f"LoRA module {module} is missing its A or B factor")
    return cfg, pairs


def merge_lora(base_dir: str | Path, adapter_dir: str | Path, out_dir: str | Path) -> dict:
    """Write `out_dir` = `base_dir` with the adapter folded into the weights. Returns counts for the log."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base_dir, adapter_dir, out_dir = Path(base_dir), Path(adapter_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, pairs = read_adapter(adapter_dir)

    index = None
    if (base_dir / INDEX_FILE).exists():
        index = json.loads((base_dir / INDEX_FILE).read_text())
        weight_map: dict[str, str] = index["weight_map"]
    elif (base_dir / SINGLE_FILE).exists():
        with safe_open(str(base_dir / SINGLE_FILE), framework="pt") as f:
            weight_map = {k: SINGLE_FILE for k in f.keys()}
    else:
        raise FileNotFoundError(f"no {INDEX_FILE} or {SINGLE_FILE} in {base_dir}")

    targets: dict[str, dict[str, tuple[str, object, object]]] = {}  # shard -> tensor name -> (module, A, B)
    for module, ab in pairs.items():
        name = next((c for c in base_tensor_candidates(module) if c in weight_map), None)
        if name is None:
            raise KeyError(f"no base weight for LoRA module {module} (tried {base_tensor_candidates(module)})")
        targets.setdefault(weight_map[name], {})[name] = (module, ab["A"], ab["B"])

    merged = 0
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        tensors = load_file(str(base_dir / shard))
        for name, (module, a, b) in targets.get(shard, {}).items():
            w = tensors[name]
            r = a.shape[0]
            if b.shape[1] != r or a.shape[1] != w.shape[1] or b.shape[0] != w.shape[0]:
                raise ValueError(f"{module}: W {tuple(w.shape)} does not match A {tuple(a.shape)} x B {tuple(b.shape)}")
            delta = (b.float() @ a.float()) * lora_scale(cfg, module, r)
            tensors[name] = (w.float() + delta).to(w.dtype).contiguous()
            merged += 1
        save_file(tensors, str(out_dir / shard), metadata={"format": "pt"})
        del tensors
    if index is not None:
        (out_dir / INDEX_FILE).write_text(json.dumps(index, indent=2))
    for f in base_dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != INDEX_FILE:
            shutil.copy2(f, out_dir / f.name)
    if (adapter_dir / "adas_vla_meta.json").exists():  # provenance of the fine-tune (train args, base model)
        shutil.copy2(adapter_dir / "adas_vla_meta.json", out_dir / "adas_vla_meta.json")
    return {"merged": merged, "lora_modules": len(pairs), "shards": len(shards)}
