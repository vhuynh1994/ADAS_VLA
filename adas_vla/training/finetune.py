"""LoRA / QLoRA supervised fine-tuning of the VLM to emit the driving-decision JSON.

Only the language model gets adapters (to learn the action vocabulary); the vision encoder stays frozen,
which keeps the ViT sub-model unchanged for NPU deployment and fits an 8 GB GPU.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from ..config import Config
from ..reasoning.prompts import decision_messages
from ..reasoning.vlm import load_model_and_processor, to_pil
from .data import load_records, target_json

# Attention + MLP projections of the language model only (anything under `visual` is excluded).
LORA_TARGET_REGEX = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


@dataclass
class TrainArgs:
    data: str
    output_dir: str = "checkpoints/lora"
    epochs: int = 3
    lr: float = 2e-4
    grad_accum: int = 8
    lora_r: int = 16
    val_data: str | None = None
    seed: int = 0
    log_every: int = 5
    max_class_share: float = 0.35  # cap each (longitudinal, lateral) label at this share of an epoch
    init_adapter: str | None = None  # continue training from an existing LoRA adapter (warm start)
    val_limit: int = 150  # val-loss subset size (full evaluation is `adas-vla eval`)


def label_key(rec: dict) -> tuple[str, str]:
    t = rec["target"]
    return t.get("longitudinal", t.get("action", "?")), t.get("lateral", "KEEP_LANE")


def balanced_epoch(records: list[dict], max_share: float, rng: random.Random) -> list[int]:
    """Indices for one epoch: every sample of minority labels, a fresh random subset of dominant ones.

    Highway data is mostly KEEP / KEEP_LANE; without a cap the model learns to always answer that.
    """
    groups: dict[tuple, list[int]] = {}
    for i, rec in enumerate(records):
        groups.setdefault(label_key(rec), []).append(i)
    if max_share >= 1 or len(groups) < 2:
        order = list(range(len(records)))
    else:
        # Solve for the per-class cap so that capped classes make up at most max_share each.
        sizes = sorted(len(v) for v in groups.values())
        cap = sizes[-1]
        while cap > 1:
            total = sum(min(n, cap) for n in sizes)
            if cap <= max_share * total:
                break
            cap -= 1
        order = [i for idx in groups.values() for i in (idx if len(idx) <= cap else rng.sample(idx, cap))]
    rng.shuffle(order)
    return order


def build_example(processor, rec: dict, cfg: Config) -> dict:
    """Tokenize one sample; loss is computed on the assistant answer only."""
    image = to_pil(Image.open(rec["image_path"]), cfg.vlm.image_max_side, cfg.vlm.image_size)
    prompt = decision_messages(image, rec["context"], rec["ego_speed_kmh"], rec["cruise_speed_kmh"],
                               cfg.vlm.language)
    full = prompt + [{"role": "assistant", "content": [{"type": "text", "text": target_json(rec["target"])}]}]
    kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    enc_full = processor.apply_chat_template(full, **kwargs)
    enc_prompt = processor.apply_chat_template(prompt, add_generation_prompt=True, **kwargs)
    n_prompt = enc_prompt["input_ids"].shape[1]
    if not bool((enc_full["input_ids"][0, :n_prompt] == enc_prompt["input_ids"][0]).all()):
        raise RuntimeError("Chat template prefix mismatch: cannot mask the prompt reliably")
    labels = enc_full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    enc_full["labels"] = labels
    return dict(enc_full)


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def train(cfg: Config, args: TrainArgs) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    all_records = load_records(args.data)
    records = [r for r in all_records if r.get("split", "train") == "train"]
    val_records = load_records(args.val_data) if args.val_data else \
        [r for r in all_records if r.get("split") == "val"]
    rng = random.Random(args.seed)
    val_records = rng.sample(val_records, min(len(val_records), args.val_limit))
    counts: dict[tuple, int] = {}
    for r in records:
        counts[label_key(r)] = counts.get(label_key(r), 0) + 1
    print(f"Train samples: {len(records)}, val-loss samples: {len(val_records)}")
    print("Train label counts:", {f"{a}/{b}": n for (a, b), n in sorted(counts.items(), key=lambda kv: -kv[1])})
    epoch_size = len(balanced_epoch(records, args.max_class_share, random.Random(0)))
    print(f"Balanced epoch size: {epoch_size} (max class share {args.max_class_share:.0%})")

    model, processor = load_model_and_processor(cfg.vlm, for_training=True)
    # QLoRA without peft's fp32 upcast of all non-quantized weights: bf16 is stable for LoRA here and
    # keeps the (unquantized) vision tower and embeddings from doubling in size on an 8 GB GPU.
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False

    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"Warm start from {args.init_adapter}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            target_modules=LORA_TARGET_REGEX, task_type="CAUSAL_LM",
        ))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, math.ceil(epoch_size * args.epochs / args.grad_accum))
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 20), total_steps)
    device = next(model.parameters()).device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step, running, t0 = 0, 0.0, time.time()
    for epoch in range(args.epochs):
        model.train()
        order = balanced_epoch(records, args.max_class_share, rng)
        for i, rec_idx in enumerate(order):
            batch = _to_device(build_example(processor, records[rec_idx], cfg), device)
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += loss.item()
            if (i + 1) % args.grad_accum == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == total_steps:
                    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                    print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running:.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} peak_mem {mem:.1f}GB "
                          f"elapsed {time.time() - t0:.0f}s", flush=True)
                running = 0.0
        if val_records:
            print(f"epoch {epoch + 1} val_loss {_val_loss(model, processor, val_records, cfg, device):.4f}")

    model.save_pretrained(out_dir)
    meta = {"base_model": cfg.vlm.model_id, "quantization": cfg.vlm.quantization,
            "image_size": cfg.vlm.image_size, "language": cfg.vlm.language, "train": asdict(args)}
    (out_dir / "adas_vla_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved LoRA adapter to {out_dir}. Use it with: --set vlm.adapter_path={out_dir}")


def _val_loss(model, processor, records, cfg, device) -> float:
    import torch

    model.eval()
    total = 0.0
    with torch.no_grad():
        for rec in records:
            total += model(**_to_device(build_example(processor, rec, cfg), device)).loss.item()
    model.train()
    return total / max(1, len(records))
