#!/usr/bin/env python3
"""Layer fusion for YOLOP's BottleneckCSP blocks: fold the BatchNorm that follows Concat(Conv, Conv) into the convs.

The PyTorch exporter folds a BatchNorm only when it directly follows a Conv, so YOLOP's CSP pattern
    Conv_a, Conv_b -> Concat -> BatchNormalization -> LeakyRelu
keeps a standalone BatchNorm (and QAIRT does not fold it either: the DLC has a Batchnorm op reading and writing the
whole concatenated tensor). BatchNorm is a per-channel affine map, so each slice of its channels belongs to one conv:

    s = gamma / sqrt(var + eps)        W'_k = W_k * s_k (per output channel)        b'_k = (b_k - mean_k) * s_k + beta_k

and the BatchNorm disappears. The LeakyRelu is elementwise too, so it moves in front of the Concat
(LeakyRelu(Concat(a, b)) = Concat(LeakyRelu(a), LeakyRelu(b))): each branch becomes Conv + LeakyRelu, which the HTP
graph compiler fuses into one op. Both rewrites are exact in real arithmetic; in FP32 the outputs agree to rounding.

  python scripts/fold_bn_concat.py models/yolop/yolop-lane-384x640.onnx models/yolop/yolop-lane-384x640-fused.onnx \
      --check outputs/deploy_sa8650/yolop_lane/eval/input_list.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def fold(model: onnx.ModelProto, move_activation: bool = True) -> tuple[onnx.ModelProto, int]:
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    graph_outputs = {o.name for o in g.output}
    remove, add, folded = set(), [], 0

    def rewire(old: str, new: str) -> None:
        for n in consumers.get(old, []):
            for k, name in enumerate(n.input):
                if name == old:
                    n.input[k] = new

    for bn in [n for n in g.node if n.op_type == "BatchNormalization"]:
        cat = producer.get(bn.input[0])
        if cat is None or cat.op_type != "Concat" or len(consumers[cat.output[0]]) != 1:
            continue
        axis = next((a.i for a in cat.attribute if a.name == "axis"), None)
        convs = [producer.get(i) for i in cat.input]
        if axis != 1 or any(c is None or c.op_type != "Conv" or len(consumers[c.output[0]]) != 1 for c in convs):
            continue
        gamma, beta, mean, var = (numpy_helper.to_array(inits[n]).astype(np.float64) for n in bn.input[1:5])
        eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-5)
        s = gamma / np.sqrt(var + eps)
        offset = 0
        for conv in convs:
            w = numpy_helper.to_array(inits[conv.input[1]]).astype(np.float64)
            cout = w.shape[0]
            sk, bk, mk = s[offset:offset + cout], beta[offset:offset + cout], mean[offset:offset + cout]
            b = numpy_helper.to_array(inits[conv.input[2]]).astype(np.float64) if len(conv.input) > 2 else np.zeros(cout)
            w_new = (w * sk.reshape(-1, *([1] * (w.ndim - 1)))).astype(np.float32)
            b_new = ((b - mk) * sk + bk).astype(np.float32)
            inits[conv.input[1]].CopyFrom(numpy_helper.from_array(w_new, conv.input[1]))
            bias_name = conv.input[2] if len(conv.input) > 2 else conv.name + "_folded_bias"
            if len(conv.input) > 2:
                inits[bias_name].CopyFrom(numpy_helper.from_array(b_new, bias_name))
            else:
                g.initializer.append(numpy_helper.from_array(b_new, bias_name))
                conv.input.append(bias_name)
            offset += cout
        if offset != len(s):
            raise ValueError(f"{bn.name}: conv channels {offset} != BatchNorm channels {len(s)}")
        remove.add(bn.name)
        act = consumers.get(bn.output[0], [])
        if (move_activation and len(act) == 1 and act[0].op_type == "LeakyRelu"
                and bn.output[0] not in graph_outputs):
            relu = act[0]
            alpha = next((a.f for a in relu.attribute if a.name == "alpha"), 0.01)
            for k, conv in enumerate(convs):  # Conv -> LeakyRelu per branch, then Concat
                out = f"{conv.output[0]}_act"
                add.append(helper.make_node("LeakyRelu", [conv.output[0]], [out], name=f"{relu.name}_{k}", alpha=alpha))
                cat.input[k] = out
            rewire(relu.output[0], cat.output[0])
            remove.add(relu.name)
        else:
            rewire(bn.output[0], cat.output[0])
        folded += 1

    nodes = [n for n in g.node if n.name not in remove]
    # keep a topological order: each new activation right after its conv
    out_nodes = []
    for n in nodes:
        out_nodes.append(n)
        out_nodes += [a for a in add if a.input[0] in n.output]
    del g.node[:]
    g.node.extend(out_nodes)
    used = {i for n in g.node for i in n.input}  # drop the folded BatchNorm parameters
    keep = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    onnx.checker.check_model(model)
    return model, folded


def check(original: str, fused: str, input_list: str, limit: int = 20) -> None:
    import onnxruntime as ort

    a = ort.InferenceSession(original, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(fused, providers=["CPUExecutionProvider"])
    inp = a.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    root = Path(input_list).parent
    files = [ln.split()[0].split(":=")[-1] for ln in Path(input_list).read_text().splitlines() if ln.strip()][:limit]
    worst, agree = 0.0, []
    for f in files:
        x = np.fromfile(root / f if not Path(f).is_absolute() else f, np.float32).reshape(shape)
        ya, yb = a.run(None, {inp.name: x}), b.run(None, {inp.name: x})
        for u, v in zip(ya, yb):
            worst = max(worst, float(np.abs(u - v).max() / (np.abs(u).max() + 1e-12)))
            if u.ndim == 4 and u.shape[1] == 2:  # segmentation: argmax agreement of the masks
                agree.append(float((u.argmax(1) == v.argmax(1)).mean()))
    print(f"FP32 parity on {len(files)} inputs: max relative difference {worst:.2e}"
          + (f", mask agreement {100 * min(agree):.4f} % (worst input)" if agree else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-activation", action="store_true", help="do not move LeakyRelu in front of the Concat")
    ap.add_argument("--check", help="input_list.txt of raw float32 inputs for the FP32 parity check")
    args = ap.parse_args()
    model = onnx.load(args.src)
    model, n = fold(model, move_activation=not args.keep_activation)
    onnx.save(model, args.dst)
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"folded {n} BatchNorm(s) after Concat; BatchNormalization left: {ops.get('BatchNormalization', 0)}, "
          f"nodes {len(model.graph.node)} -> {args.dst}")
    if args.check:
        check(args.src, args.dst, args.check)


if __name__ == "__main__":
    main()
