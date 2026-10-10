"""Shard-wise LoRA merge (training/merge.py). Needs torch + safetensors (skipped in the minimal CI environment)."""

import json

import pytest

from adas_vla.training.merge import base_tensor_candidates, lora_scale

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")


def _bf16(*shape):
    return torch.randn(*shape).to(torch.bfloat16)


def _write_base(base, shards: dict[str, dict], extra_files=("config.json", "preprocessor_config.json")):
    from safetensors.torch import save_file

    base.mkdir()
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, str(base / shard), metadata={"format": "pt"})
        weight_map.update({k: shard for k in tensors})
    if len(shards) > 1:
        (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}))
    for name in extra_files:
        (base / name).write_text("{}")


def _write_adapter(adapter, tensors: dict, r: int = 2, alpha: int = 4, **cfg):
    from safetensors.torch import save_file

    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": alpha,
                                                             "bias": "none", **cfg}))
    save_file(tensors, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
    (adapter / "adas_vla_meta.json").write_text('{"base_model": "test"}')


def test_merge_lora_shards_matches_the_closed_form(tmp_path):
    from safetensors.torch import load_file

    from adas_vla.training.merge import merge_lora

    w_q, w_k, w_norm, w_vis = _bf16(8, 6), _bf16(8, 6), _bf16(4), _bf16(5, 5)
    _write_base(tmp_path / "base", {
        "model-00001-of-00002.safetensors": {"model.layers.0.self_attn.q_proj.weight": w_q,
                                             "model.layers.0.self_attn.k_proj.weight": w_k},
        "model-00002-of-00002.safetensors": {"model.norm.weight": w_norm, "visual.blocks.0.attn.qkv.weight": w_vis},
    })
    a, b = _bf16(2, 6), _bf16(8, 2)
    # peft names the module as transformers exposes it at load time (model.language_model.*), the file as model.*
    _write_adapter(tmp_path / "lora", {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": b,
    })
    stats = merge_lora(tmp_path / "base", tmp_path / "lora", tmp_path / "out")
    assert stats == {"merged": 1, "lora_modules": 1, "shards": 2}
    out1 = load_file(str(tmp_path / "out" / "model-00001-of-00002.safetensors"))
    expected = (w_q.float() + 2.0 * (b.float() @ a.float())).to(torch.bfloat16)  # alpha / r = 4 / 2
    assert torch.equal(out1["model.layers.0.self_attn.q_proj.weight"], expected)
    assert torch.equal(out1["model.layers.0.self_attn.k_proj.weight"], w_k)
    out2 = load_file(str(tmp_path / "out" / "model-00002-of-00002.safetensors"))
    assert torch.equal(out2["visual.blocks.0.attn.qkv.weight"], w_vis)
    out = tmp_path / "out"
    assert json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]["model.norm.weight"].endswith("00002.safetensors")
    assert (out / "config.json").exists() and (out / "preprocessor_config.json").exists()
    assert (out / "adas_vla_meta.json").exists()


def test_merge_single_file_base_and_rslora(tmp_path):
    from safetensors.torch import load_file

    from adas_vla.training.merge import merge_lora

    w = _bf16(4, 3)
    _write_base(tmp_path / "base", {"model.safetensors": {"model.layers.0.mlp.up_proj.weight": w}})
    a, b = _bf16(4, 3), _bf16(4, 4)
    _write_adapter(tmp_path / "lora", {"base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": a,
                                       "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": b},
                   r=4, alpha=8, use_rslora=True)
    merge_lora(tmp_path / "base", tmp_path / "lora", tmp_path / "out")
    got = load_file(str(tmp_path / "out" / "model.safetensors"))["model.layers.0.mlp.up_proj.weight"]
    expected = (w.float() + (8 / 2.0) * (b.float() @ a.float())).to(torch.bfloat16)  # alpha / sqrt(r)
    assert torch.equal(got, expected)


def test_merge_refuses_unknown_modules_and_shapes(tmp_path):
    from adas_vla.training.merge import merge_lora

    _write_base(tmp_path / "base", {"model.safetensors": {"model.layers.0.self_attn.q_proj.weight": _bf16(8, 6)}})
    _write_adapter(tmp_path / "lora", {"base_model.model.model.layers.9.self_attn.q_proj.lora_A.weight": _bf16(2, 6),
                                       "base_model.model.model.layers.9.self_attn.q_proj.lora_B.weight": _bf16(8, 2)})
    with pytest.raises(KeyError):
        merge_lora(tmp_path / "base", tmp_path / "lora", tmp_path / "out")
    _write_adapter(tmp_path / "lora2", {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": _bf16(2, 5),
                                        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": _bf16(8, 2)})
    with pytest.raises(ValueError):
        merge_lora(tmp_path / "base", tmp_path / "lora2", tmp_path / "out2")
    _write_adapter(tmp_path / "lora3", {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": _bf16(2, 6),
                                        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": _bf16(8, 2)},
                   modules_to_save=["lm_head"])
    with pytest.raises(NotImplementedError):
        merge_lora(tmp_path / "base", tmp_path / "lora3", tmp_path / "out3")


def test_candidates_and_scale_helpers():
    assert base_tensor_candidates("model.language_model.layers.3.mlp.down_proj") == [
        "model.language_model.layers.3.mlp.down_proj.weight", "model.layers.3.mlp.down_proj.weight"]
    assert base_tensor_candidates("model.layers.3.mlp.down_proj") == ["model.layers.3.mlp.down_proj.weight"]
    assert lora_scale({"lora_alpha": 32}, "x", 16) == 2.0
    assert lora_scale({"lora_alpha": 32, "alpha_pattern": {"down_proj": 64}}, "model.layers.0.mlp.down_proj", 16) == 4.0
