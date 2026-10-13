"""InternVL3.5-2B (transformers-native checkpoint) wired into the decision prompt, training and merge code.

Needs models/OpenGVLab--InternVL3_5-2B-HF (scripts/fetch_hf.py OpenGVLab/InternVL3_5-2B-HF); skipped otherwise.
Only the processor, tokenizer and the checkpoint's tensor names are read: no weights are loaded.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("PIL")

from adas_vla.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "OpenGVLab--InternVL3_5-2B-HF"
pytestmark = pytest.mark.skipif(not (MODEL / "model.safetensors").exists(), reason="InternVL3.5-2B-HF not downloaded")


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(MODEL))


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "vlm_internvl.yaml")


def _record(tmp_path):
    from PIL import Image

    img = tmp_path / "frame.jpg"
    Image.new("RGB", (1280, 720), (90, 120, 150)).save(img)  # a 16:9 dashcam frame
    return {"image_path": str(img), "image_prev_path": None, "context": "Lead: car ahead at 22 m, closing 1.2 m/s.",
            "ego_speed_kmh": 50.0, "cruise_speed_kmh": 60.0,
            "target": {"longitudinal": "DECELERATE", "lateral": "KEEP_LANE", "target_speed_kmh": 40, "risk": "medium",
                       "reason": "Car ahead slowing."}}


def test_one_448_tile_is_256_image_tokens_like_the_board(processor, cfg, tmp_path):
    from PIL import Image

    from adas_vla.reasoning.prompts import decision_messages
    from adas_vla.reasoning.vlm import encode_messages, to_pil

    image = to_pil(Image.open(_record(tmp_path)["image_path"]), cfg.vlm.image_max_side, cfg.vlm.image_size)
    assert image.size == (448, 448)
    enc = encode_messages(processor, decision_messages(image, "ctx", 50.0, 60.0, cfg.vlm.language), True)
    assert tuple(enc["pixel_values"].shape) == (1, 3, 448, 448)  # one tile, no thumbnail
    assert int((enc["input_ids"] == processor.image_token_id).sum()) == 256


def test_training_example_masks_the_prompt(processor, cfg, tmp_path):
    from adas_vla.training.data import target_json
    from adas_vla.training.finetune import build_example

    rec = _record(tmp_path)
    ex = build_example(processor, rec, cfg)  # raises on a chat-template prefix mismatch
    answer = ex["labels"][0][ex["labels"][0] != -100]
    text = processor.tokenizer.decode(answer, skip_special_tokens=True).strip()
    assert text.startswith(target_json(rec["target"])[:40])


def test_actions_have_distinct_first_tokens(processor):
    from adas_vla.reasoning.vlm import LONG_ORDER

    tok = processor.tokenizer
    firsts = {a: tok(a, add_special_tokens=False).input_ids[0] for a in LONG_ORDER}
    assert len(set(firsts.values())) == len(firsts), firsts


def test_lora_modules_map_to_checkpoint_tensors():
    import re

    from safetensors import safe_open

    from adas_vla.training.finetune import LORA_TARGET_REGEX
    from adas_vla.training.merge import base_tensor_candidates

    with safe_open(str(MODEL / "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    n_layers = json.loads((MODEL / "config.json").read_text())["text_config"]["num_hidden_layers"]
    modules = [f"model.language_model.layers.{i}.{part}" for i in range(n_layers)
               for part in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    assert all(re.match(LORA_TARGET_REGEX, m) for m in modules)
    missing = [m for m in modules if not any(c in keys for c in base_tensor_candidates(m))]
    assert not missing, missing[:3]
