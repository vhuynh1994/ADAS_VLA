"""GPTQ (Frantar et al., 2022) 4-bit weights for the LLM of the deployment form (deploy/qwen_vl.py), on an 8 GB GPU.

Round-to-nearest 4-bit weights per output channel cost v3 ~20 points of joint accuracy even with the Hadamard
rotation (scripts/v3_qnn.py eval --rotate --wbits 4). GPTQ quantizes one layer at a time from the activations
of real decision prompts: each weight column is rounded in turn and the error is pushed onto the columns not yet
rounded through the inverse Hessian of the layer input (H = X^T X), so the layer output, not the weights, stays
close. The grids are fixed before (symmetric, per output channel or LPBQ blocks: what the HTP executes), so the
result is exactly representable and goes to QAIRT as encoding overrides.

Order: the float layer l sees the hidden states produced by the already-quantized layers 0..l-1 (sequential), all
projections of a layer get their Hessians from the float layer's inputs. Columns in descending Hessian diagonal
order (act-order); damping 1 % of the mean diagonal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import qwen_vl as qv

PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")


def grid_scales(w: torch.Tensor, bits: int, block: int | None, clip_search: bool = True) -> torch.Tensor:
    """Per-weight scale grid [out, in] (constant per row, or per row and block of `block` inputs: LPBQ, block scale =
    channel scale x integer 1..16). With `clip_search` each row's (or block's) range is shrunk by the factor in
    0.70..1.00 that minimises the squared rounding error (round-to-nearest, before GPTQ)."""
    qmax = 2 ** (bits - 1) - 1
    o, i = w.shape
    g = block or i
    wb = w.reshape(o, i // g, g)
    amax = wb.abs().amax(2, keepdim=True).clamp_min(1e-12)
    best, best_err = amax / qmax, None
    if clip_search:
        for r in torch.linspace(1.0, 0.70, 7):
            s = amax * r / qmax
            err = ((wb - (wb / s).round().clamp(-qmax - 1, qmax) * s) ** 2).sum(2, keepdim=True)
            if best_err is None:
                best_err = err
            better = err < best_err
            best, best_err = torch.where(better, s, best), torch.where(better, err, best_err)
    s = best
    if block is not None:  # LPBQ: the block scales are integer multiples (1..16) of one scale per channel
        sc = s.amax(1, keepdim=True) / 16
        s = (s / sc).round().clamp(1, 16) * sc
    return s.expand(o, i // g, g).reshape(o, i)


def gptq(w: torch.Tensor, h: torch.Tensor, scales: torch.Tensor, bits: int, block_cols: int = 128,
         damp: float = 0.01) -> torch.Tensor:
    """Quantize w [out, in] (float32, GPU) on the fixed grid `scales` [out, in] using the input Hessian h [in, in];
    returns the dequantized weights (exactly on the grid)."""
    qmax = 2 ** (bits - 1) - 1
    w, h = w.clone().float(), h.clone().float()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    perm = torch.argsort(torch.diag(h), descending=True)
    w, h, s = w[:, perm], h[perm][:, perm], scales[:, perm]
    h += damp * torch.mean(torch.diag(h)) * torch.eye(h.shape[0], device=h.device)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    q = torch.zeros_like(w)
    for a in range(0, w.shape[1], block_cols):
        b = min(a + block_cols, w.shape[1])
        w1, err1, hinv1 = w[:, a:b].clone(), torch.zeros_like(w[:, a:b]), hinv[a:b, a:b]
        for j in range(b - a):
            col, sj = w1[:, j], s[:, a + j]
            qcol = (col / sj).round().clamp(-qmax - 1, qmax) * sj
            q[:, a + j] = qcol
            e = (col - qcol) / hinv1[j, j]
            w1[:, j:] -= e[:, None] * hinv1[j, j:][None, :]
            err1[:, j] = e
        w[:, b:] -= err1 @ hinv[a:b, b:]
    return q[:, torch.argsort(perm)]


def layer_io(layer: qv.DecoderLayer, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """One prompt through a layer (no KV cache): the output and the inputs of its projections
    (qkv: normed input, o: attention output, mlp: normed input, down: gated activation), all [T, C]."""
    hd = layer.q[0].out_channels
    empty_k = x.new_zeros(len(layer.k), 1, hd, 0)
    empty_v = x.new_zeros(len(layer.v), 1, 0, hd)
    h1 = qv.rms_norm(x, layer.ln1, layer.eps)
    attn = _attention(layer, h1.unsqueeze(1), mask, cos, sin, empty_k, empty_v)
    mid = x + layer.o(attn).squeeze(1)
    h2 = qv.rms_norm(mid, layer.ln2, layer.eps)
    act = F.silu(layer.gate(h2)) * layer.up(h2)
    out = mid + layer.down(act)
    flat = lambda t: t.reshape(-1, t.shape[-1])  # noqa: E731
    return out, {"qkv": flat(h1), "o": flat(attn), "mlp": flat(h2), "down": flat(act)}


def _attention(layer, h, mask, cos, sin, past_k, past_v):
    q = [qv.rope_half(p(h), cos, sin) * layer.scale for p in layer.q]
    k = [qv.rope_half(p(h), cos, sin) for p in layer.k]
    if layer.r3 is not None:
        q, k = [t @ layer.r3 for t in q], [t @ layer.r3 for t in k]
    v = [p(h) for p in layer.v]
    out = []
    for j in range(len(k)):
        keys = torch.cat([past_k[j:j + 1], k[j].transpose(2, 3)], 3)
        values = torch.cat([past_v[j:j + 1], v[j]], 2)
        for qi in q[j * layer.group:(j + 1) * layer.group]:
            out.append(torch.softmax(qi @ keys + mask, -1) @ values)
    return torch.cat(out, -1)


def convs_of(layer: qv.DecoderLayer) -> dict[str, list]:
    return {"q": list(layer.q), "k": list(layer.k), "v": list(layer.v), "o": [layer.o], "gate": [layer.gate],
            "up": [layer.up], "down": [layer.down]}


HESSIAN_OF = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "mlp", "up": "mlp", "down": "down"}


@torch.no_grad()
def quantize_layer(layer: qv.DecoderLayer, prompts: list[tuple], bits: int, block: int | None,
                   damp: float = 0.01) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    """GPTQ one layer in place (float32 on the GPU). prompts: [(x [1, T, C], mask, cos, sin)] -> (outputs of the
    quantized layer per prompt, per-projection scale grids [out, in], relative output error of each projection)."""
    hs: dict[str, torch.Tensor] = {}
    for x, mask, cos, sin in prompts:
        _, ins = layer_io(layer, x, mask, cos, sin)
        for k, t in ins.items():
            t = t.float()
            hs[k] = hs.get(k, 0) + t.T @ t
    grids, errors = {}, {}
    for name, convs in convs_of(layer).items():
        w = torch.cat([c.weight[:, :, 0, 0] for c in convs]).float()
        h = hs[HESSIAN_OF[name]]
        s = grid_scales(w, bits, block if w.shape[1] % (block or 1) == 0 else None)
        wq = gptq(w, h, s, bits, damp=damp)
        rtn = (w / s).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s
        err = lambda d: float(torch.einsum("oi,ij,oj->", d, h, d) / torch.einsum("oi,ij,oj->", w, h, w))  # noqa
        errors[name] = (err(w - wq), err(w - rtn))
        a = 0
        for c in convs:
            n = c.weight.shape[0]
            c.weight[:, :, 0, 0] = wq[a:a + n].to(c.weight.dtype)
            a += n
        grids[name] = s
    outs = [layer_io(layer, x, mask, cos, sin)[0] for x, mask, cos, sin in prompts]
    return outs, grids, errors


def causal_inputs(spec: qv.Spec, pos3: torch.Tensor, device, dtype=torch.float32):
    """Mask [1, 1, T, T] (causal, qwen_vl.LLM_MASK) and M-RoPE cos/sin [1, 1, T, hd/2] of a full prompt."""
    t = pos3.shape[1]
    mask = torch.triu(torch.full((t, t), qv.LLM_MASK, dtype=dtype, device=device), 1)[None, None]
    cos, sin = qv.mrope_cos_sin(pos3, spec)
    return mask, cos[None, None].to(device, dtype), sin[None, None].to(device, dtype)


def lpbq_int_scales(scales: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(per-channel float scale [out], per-block integer scales [out, in / block]) of an LPBQ grid."""
    per_block = scales[:, ::block]
    sc = per_block.amax(1) / 16
    return sc, (per_block / sc[:, None]).round().to(torch.int32)


def check_grid(w: torch.Tensor, scales: torch.Tensor, bits: int) -> float:
    """Largest distance of w / scales from an integer in the 4-bit range (0 when exactly representable)."""
    r = w.float() / scales
    qmax = 2 ** (bits - 1) - 1
    return float(max((r - r.round()).abs().max(), (r.round().clamp(-qmax - 1, qmax) - r.round()).abs().max()))



def save_layer(tensors: dict[str, torch.Tensor], layer: int, convs: dict[str, list], grids: dict[str, torch.Tensor],
               bits: int, block: int | None) -> None:
    """Integer codes (int8 [out, in]) and compact scales ([out, groups]) of one quantized layer into `tensors`."""
    for name, cs in convs.items():
        w = torch.cat([c.weight[:, :, 0, 0] for c in cs]).float()
        s = grids[name]
        g = block if block and w.shape[1] % block == 0 else w.shape[1]
        tensors[f"layers.{layer}.{name}.codes"] = (w / s).round().to(torch.int8).cpu()
        tensors[f"layers.{layer}.{name}.scales"] = s[:, ::g].contiguous().cpu()


class QuantizedWeights:
    """GPTQ result file (safetensors): dequantized weights of each layer, exactly on their 4-bit grid."""

    def __init__(self, path):
        import json

        from safetensors import safe_open

        self.f = safe_open(str(path), "pt")
        self.meta = json.loads(self.f.metadata().get("gptq", "{}"))

    def scales(self, layer: int, name: str) -> torch.Tensor:
        """Full scale grid [out, in] of one projection."""
        codes = self.f.get_slice(f"layers.{layer}.{name}.codes")
        s = self.f.get_tensor(f"layers.{layer}.{name}.scales")
        return s.repeat_interleave(codes.get_shape()[1] // s.shape[1], dim=1)

    def layer(self, layer: int, dtype=torch.float32) -> dict[str, torch.Tensor]:
        return {n: (self.f.get_tensor(f"layers.{layer}.{n}.codes").double() * self.scales(layer, n).double()).to(dtype)
                for n in PROJECTIONS}
