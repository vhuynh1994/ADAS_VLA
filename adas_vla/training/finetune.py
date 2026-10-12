"""LoRA / QLoRA supervised fine-tuning of the VLM to emit the driving-decision JSON.

Only the language model gets adapters (to learn the action vocabulary); the vision encoder stays frozen,
which keeps the ViT sub-model unchanged for NPU deployment and fits an 8 GB GPU.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Config
from ..reasoning.prompts import decision_messages
from ..reasoning.vlm import encode_messages, load_model_and_processor
from ..types import LongAction
from .data import load_records, sample_visual, target_json

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
    workers: int = 2  # DataLoader workers: decode images and tokenize while the GPU trains (0 = main thread)
    brake_weight: float = 1.0  # loss multiplier for samples labelled DECELERATE / BRAKE / STOP (targets under-braking)
    optimizer: str = "adamw"  # adamw | paged_adamw_8bit (bitsandbytes: 8-bit, paged optimizer state; 7B on 8 GB)
    max_steps: int | None = None  # stop after this many optimizer steps (smoke test: peak memory + ETA of the full run)
    save_every: int = 0  # write a resumable checkpoint (<output_dir>/resume/state.pt) every N optimizer steps (0 = off)
    resume: bool = False  # continue from <output_dir>/resume/state.pt if it exists (same run settings required)


# Settings that must match between a checkpoint and the run resuming it.
RESUME_KEYS = ("data", "epochs", "lr", "grad_accum", "lora_r", "seed", "max_class_share", "init_adapter",
               "brake_weight", "optimizer")


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


def sample_weight(rec: dict, brake_weight: float) -> float:
    """Up-weight samples whose label asks to slow down or brake: missing those is the under-braking error."""
    try:
        braking = LongAction(label_key(rec)[0]).rank >= LongAction.DECELERATE.rank
    except ValueError:
        braking = False
    return brake_weight if braking else 1.0


def build_example(processor, rec: dict, cfg: Config) -> dict:
    """Tokenize one sample; loss is computed on the assistant answer only."""
    prompt = decision_messages(sample_visual(rec, cfg), rec["context"], rec["ego_speed_kmh"],
                               rec["cruise_speed_kmh"], cfg.vlm.language)
    full = prompt + [{"role": "assistant", "content": [{"type": "text", "text": target_json(rec["target"])}]}]
    enc_full = encode_messages(processor, full, add_generation_prompt=False)
    enc_prompt = encode_messages(processor, prompt, add_generation_prompt=True)
    n_prompt = enc_prompt["input_ids"].shape[1]
    if not bool((enc_full["input_ids"][0, :n_prompt] == enc_prompt["input_ids"][0]).all()):
        raise RuntimeError("Chat template prefix mismatch: cannot mask the prompt reliably")
    labels = enc_full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    enc_full["labels"] = labels
    return dict(enc_full)


def build_optimizer(params, name: str, lr: float):
    """AdamW, or bitsandbytes' 8-bit paged AdamW (the optimizer state of a 7B LoRA shrinks ~4x and can spill to
    pinned CPU memory under pressure instead of raising an out-of-memory error)."""
    import torch

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name == "paged_adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    raise ValueError(f"Unknown optimizer: {name}")


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


MAX_OOM_SKIPS = 20  # more than this many out-of-memory windows in one run is a setup problem, not a long sample


def _micro_step(model, example: dict, device, weight: float, grad_accum: int) -> float | None:
    """Forward + backward of one sample; returns its (scaled) loss, or None when the GPU ran out of memory.

    The 7B on the 8 GB card has ~0.4 GB of headroom: the longest sample of ds_v3 (314 text tokens vs 109 median) made
    the 2026-10-11 run fail at step 1458. The caller then drops the accumulation window and goes on. Nothing is
    cleaned up inside the except block, so the traceback and the tensors it references are released first."""
    import torch

    try:
        loss = model(**_to_device(example, device)).loss * weight / grad_accum
        loss.backward()
        return loss.item()
    except torch.OutOfMemoryError:
        pass
    return None


def save_resume_state(path: Path, *, model, optimizer, scheduler, step: int, epoch: int, next_i: int,
                      order: list[int], rng: random.Random, elapsed: float, run: dict) -> None:
    """Everything needed to continue a run exactly where it stopped, written atomically (tmp file + rename)."""
    import torch

    state = {
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "step": step, "epoch": epoch, "next_i": next_i, "order": order,
        "rng": rng.getstate(), "py_random": random.getstate(), "torch_rng": torch.get_rng_state(),
        # only when this process uses CUDA (training always does): querying it would otherwise create a context
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None,
        "elapsed": elapsed, "run": run,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume_state(path: Path, *, model, optimizer, scheduler, rng: random.Random, run: dict) -> dict:
    """Restore a save_resume_state() checkpoint into a freshly built model / optimizer / scheduler; returns the
    loop position (step, epoch, next_i, order, elapsed). Refuses a checkpoint made with other run settings."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    diff = {k: (state["run"].get(k), run.get(k)) for k in run if state["run"].get(k) != run.get(k)}
    if diff:
        raise SystemExit(f"{path} was written by a run with other settings {diff}; delete it or match them")
    params = dict(model.named_parameters())
    missing = [n for n in state["trainable"] if n not in params]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} trainable tensors not in the model (e.g. {missing[0]})")
    with torch.no_grad():
        for n, t in state["trainable"].items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng.setstate(state["rng"])
    random.setstate(state["py_random"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return {k: state[k] for k in ("step", "epoch", "next_i", "order", "elapsed")}


class _Examples:
    """Map-style dataset over one epoch order; DataLoader workers do the image decoding and tokenization."""

    def __init__(self, processor, records: list[dict], order: list[int], cfg: Config):
        self.processor, self.records, self.order, self.cfg = processor, records, order, cfg

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, i: int) -> dict:
        return build_example(self.processor, self.records[self.order[i]], self.cfg)


def _identity(x):
    return x


def _loader(processor, records: list[dict], order: list[int], cfg: Config, workers: int):
    from torch.utils.data import DataLoader

    # batch_size=None: one pre-tokenized sample per step (batch dim 1), prepared ahead of the GPU by the workers.
    return DataLoader(_Examples(processor, records, order, cfg), batch_size=None, shuffle=False,
                      num_workers=workers, collate_fn=_identity, prefetch_factor=4 if workers else None)


def train(cfg: Config, args: TrainArgs) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # forked DataLoader workers + fast tokenizer
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
    print(f"Balanced epoch size: {epoch_size} (max class share {args.max_class_share:.0%}), "
          f"brake weight {args.brake_weight:g}, {args.workers} loader workers, "
          f"{'2-frame' if cfg.vlm.prev_frame_s > 0 else 'single-frame'} input")

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
    optimizer = build_optimizer(params, args.optimizer, args.lr)
    total_steps = max(1, math.ceil(epoch_size * args.epochs / args.grad_accum))
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 20), total_steps)
    device = next(model.parameters()).device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "resume" / "state.pt"
    run = {**{k: getattr(args, k) for k in RESUME_KEYS}, "model": cfg.vlm.model_id,
           "quantize_vision": cfg.vlm.quantize_vision, "prev_frame_s": cfg.vlm.prev_frame_s}
    step, running, t0, n_oom = 0, 0.0, time.time(), 0
    start_epoch, start_i, resume_order = 0, 0, None
    if args.resume and resume_path.exists():
        pos = load_resume_state(resume_path, model=model, optimizer=optimizer, scheduler=scheduler, rng=rng, run=run)
        step, start_epoch, start_i, resume_order = pos["step"], pos["epoch"], pos["next_i"], pos["order"]
        t0 -= pos["elapsed"]
        print(f"Resumed from {resume_path}: step {step}/{total_steps}, epoch {start_epoch + 1}, "
              f"sample {start_i}/{len(resume_order)}", flush=True)
    elif args.resume:
        print(f"No checkpoint at {resume_path}: starting from scratch", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if resume_order is not None:  # the interrupted epoch: same order, rng already past it
            order, resume_order = resume_order, None
        else:
            order = balanced_epoch(records, args.max_class_share, rng)
        first = start_i if epoch == start_epoch else 0
        rest = order[first:]
        window_ok = 0  # samples whose gradients are in the current accumulation window
        for j, (rec_idx, example) in enumerate(zip(rest, _loader(processor, records, rest, cfg, args.workers))):
            i = first + j
            loss = _micro_step(model, example, device, sample_weight(records[rec_idx], args.brake_weight),
                               args.grad_accum)
            if loss is None:  # out of GPU memory: the window's gradients may be partial, drop them all
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                window_ok, running, n_oom = 0, 0.0, n_oom + 1
                print(f"out of GPU memory on sample {records[rec_idx].get('id', rec_idx)} (epoch {epoch + 1}, "
                      f"step {step + 1}): dropped this accumulation window ({n_oom} so far)", flush=True)
                if n_oom > MAX_OOM_SKIPS:
                    raise SystemExit(f"{n_oom} out-of-memory windows: reduce memory (lora_r, quantize_vision, ...)")
            else:
                running += loss
                window_ok += 1
            if (i + 1) % args.grad_accum == 0 or i == len(order) - 1:
                if window_ok:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                scheduler.step()  # the schedule and the step count stay aligned with total_steps (and resume)
                optimizer.zero_grad(set_to_none=True)
                window_ok = 0
                step += 1
                if step % args.log_every == 0 or step == total_steps or step == args.max_steps:
                    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                    elapsed = time.time() - t0
                    print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running:.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} peak_mem {mem:.1f}GB "
                          f"elapsed {elapsed:.0f}s eta {elapsed / step * (total_steps - step) / 60:.0f}min", flush=True)
                running = 0.0
                if args.save_every and step % args.save_every == 0:
                    save_resume_state(resume_path, model=model, optimizer=optimizer, scheduler=scheduler, step=step,
                                      epoch=epoch, next_i=i + 1, order=order, rng=rng, elapsed=time.time() - t0,
                                      run=run)
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            print(f"Stopped after {step} optimizer steps (--max-steps); the full run would take "
                  f"{(time.time() - t0) / step * total_steps / 3600:.1f} h at this pace.")
            break
        if val_records:
            print(f"epoch {epoch + 1} val_loss {_val_loss(model, processor, val_records, cfg, device):.4f}")

    model.save_pretrained(out_dir)
    meta = {"base_model": cfg.vlm.model_id, "quantization": cfg.vlm.quantization,
            "image_size": cfg.vlm.image_size, "language": cfg.vlm.language, "train": asdict(args)}
    (out_dir / "adas_vla_meta.json").write_text(json.dumps(meta, indent=2))
    if not (args.max_steps and step >= args.max_steps) and resume_path.exists():
        shutil.rmtree(resume_path.parent)  # finished: a later --resume must not continue a completed run
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
