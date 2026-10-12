"""Quantization overrides that keep a region of an ONNX graph in FP16 (mixed precision on the HTP).

  python scripts/qnn_fp16_overrides.py model.onnx out.json --region '^/model\\.(8|9|10|20|21|22)/|^/model\\.23/cv[23]\\.2/' \\
      [--base attention_overrides.json]
  qairt-converter --input_network model.onnx --quantization_overrides out.json --output_path model_ov.dlc
  qairt-quantizer --input_dlc model_ov.dlc --output_dlc model_q.dlc --input_list calib.txt --act_bitwidth 16 ...

Every float output of the nodes whose name matches --region gets {"bitwidth": 16, "dtype": "float"}; the rest of the
graph keeps the quantizer's encodings. QAIRT 2.46 takes overrides at conversion time, not in qairt-quantizer.
YOLO11s W8A16 for the SA8650P: the stride-32 path above in FP16 (131 tensors) recovered 105/106 float detections vs
84/106 for W8A8 (see CLAUDE.md, 2026-10-12).
"""

from __future__ import annotations

import argparse
import json
import re


def main() -> None:
    import onnx

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("onnx")
    p.add_argument("out")
    p.add_argument("--region", required=True, help="regex on node names")
    p.add_argument("--base", help="existing overrides to extend (e.g. symmetric int16 attention encodings)")
    a = p.parse_args()
    m = onnx.shape_inference.infer_shapes(onnx.load(a.onnx))
    types = {vi.name: vi.type.tensor_type.elem_type for vi in list(m.graph.value_info) + list(m.graph.output)}
    region = re.compile(a.region)
    names = [o for n in m.graph.node if region.search(n.name or "") for o in n.output
             if o and types.get(o) == onnx.TensorProto.FLOAT]
    ov = json.load(open(a.base)) if a.base else {"activation_encodings": {}, "param_encodings": {}}
    for t in names:
        ov["activation_encodings"][t] = [{"bitwidth": 16, "dtype": "float"}]
    with open(a.out, "w") as fh:
        json.dump(ov, fh, indent=1)
    print(f"{a.out}: {len(names)} tensors in FP16")


if __name__ == "__main__":
    main()
