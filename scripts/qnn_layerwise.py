"""Layer-wise SQNR of a quantized QNN model against the float ONNX model, to find where quantization breaks.

  qnn-net-run --backend libQnnHtp.so --model libQnnModelDlc.so --dlc_path model_q.dlc --input_list one.txt \\
      --output_dir dump_q --debug --config_file htp_config.json          # HTP emulator, every tensor dumped
  python scripts/qnn_layerwise.py model.onnx input.raw 1,3,384,640 out.tsv q=dump_q/Result_0 [q2=...]

Each float tensor of the ONNX graph (ONNX Runtime) is compared with the dump of the same name (QNN replaces every
non-alphanumeric character by '_'; 4-D activations are dumped NHWC, so both layouts are tried). SQNR =
10 log10(|float|^2 / |float - quant|^2) in dB; a tensor whose SQNR is far below its inputs' is where it breaks.
"""

from __future__ import annotations

import re
import sys

import numpy as np


def sqnr(f: np.ndarray, q: np.ndarray) -> float | None:
    if q.size != f.size:
        return None
    cands = [q.reshape(f.shape)]
    if f.ndim == 4:
        n, c, h, w = f.shape
        cands.append(q.reshape(n, h, w, c).transpose(0, 3, 1, 2))
    best = None
    for cand in cands:
        err = float(np.sum((f - cand) ** 2))
        s = 10 * np.log10(float(np.sum(f ** 2)) / err) if err > 0 else 99.0
        best = s if best is None else max(best, s)
    return best


def main() -> None:
    import onnx
    import onnxruntime as ort
    from pathlib import Path

    onnx_path, raw, shape, out = sys.argv[1:5]
    dumps = dict(a.split("=", 1) for a in sys.argv[5:])
    m = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    types = {vi.name: vi.type.tensor_type.elem_type for vi in list(m.graph.value_info) + list(m.graph.output)}
    order = [o for n in m.graph.node for o in n.output if o and types.get(o) == onnx.TensorProto.FLOAT]
    del m.graph.output[:]
    for o in order:
        m.graph.output.append(onnx.helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.fromfile(raw, np.float32).reshape([int(v) for v in shape.split(",")])
    vals = dict(zip(order, sess.run(order, {sess.get_inputs()[0].name: x})))
    with open(out, "w") as fh:
        fh.write("tensor\tshape\t" + "\t".join(dumps) + "\n")
        for o in order:
            cols = []
            for d in dumps.values():
                p = Path(d) / (re.sub(r"[^0-9A-Za-z]", "_", o) + ".raw")
                s = sqnr(vals[o], np.fromfile(p, np.float32)) if p.exists() else None
                cols.append("-" if s is None else f"{s:.1f}")
            fh.write(f"{o}\t{tuple(vals[o].shape)}\t" + "\t".join(cols) + "\n")
    print(f"{len(order)} tensors -> {out}")


if __name__ == "__main__":
    main()
