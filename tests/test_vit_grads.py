import copy
import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from adas_vla.training.finetune import LORA_TARGET_REGEX, freeze_vision_autograd  # noqa: E402

IMAGE, VSTART, VEND = 150, 152, 153


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda *a, **k: None)


def _tiny_qwen25_vl():
    torch.manual_seed(0)
    cfg = transformers.Qwen2_5_VLConfig(
        text_config=dict(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}, bos_token_id=1, eos_token_id=2),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=64,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=56,
                           fullatt_block_indexes=[1], in_channels=3),
        image_token_id=IMAGE, video_token_id=151, vision_start_token_id=VSTART, vision_end_token_id=VEND)
    model = transformers.Qwen2_5_VLForConditionalGeneration(cfg)
    ids = torch.tensor([[5, 6, VSTART, IMAGE, IMAGE, IMAGE, IMAGE, VEND, 7, 8, 9, 10]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids,
                  pixel_values=torch.randn(16, 3 * 2 * 14 * 14), image_grid_thw=torch.tensor([[1, 4, 4]]))
    return model, inputs, lambda m: m.base_model.model.model.visual.merger


def _tiny_internvl():
    """InternVL (transformers-native, like InternVL3.5-2B-HF): 56x56 image -> 16 patches -> 4 tokens."""
    torch.manual_seed(0)
    cfg = transformers.InternVLConfig(
        vision_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                           image_size=56, patch_size=14),
        text_config=dict(model_type="qwen3", vocab_size=200, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=512, bos_token_id=1, eos_token_id=2),
        image_token_id=IMAGE, image_seq_length=4)
    model = transformers.InternVLForConditionalGeneration(cfg)
    ids = torch.tensor([[5, 6, VSTART, IMAGE, IMAGE, IMAGE, IMAGE, VEND, 7, 8, 9, 10]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, pixel_values=torch.randn(1, 3, 56, 56))
    return model, inputs, lambda m: m.base_model.model.model.multi_modal_projector


def _lora_grads(model, inputs, image_features, input_hook) -> tuple[dict, bool]:
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(1)  # same LoRA init in both runs
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGET_REGEX,
                                                       init_lora_weights=False))
    input_hook(model)
    model.train()
    vit_grad = []  # do the image features leave the vision part with an autograd graph?
    image_features(model).register_forward_hook(lambda m, i, o: vit_grad.append(o.requires_grad))
    torch.manual_seed(2)
    model(**inputs).loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    return grads, vit_grad[0]


@pytest.mark.parametrize("build", [_tiny_qwen25_vl, _tiny_internvl], ids=["qwen2.5-vl", "internvl"])
def test_lora_gradients_without_backprop_through_the_frozen_vit(build):
    base, inputs, image_features = build()
    ref, ref_vit = _lora_grads(copy.deepcopy(base), inputs, image_features, lambda m: None)
    new, new_vit = _lora_grads(copy.deepcopy(base), inputs, image_features, freeze_vision_autograd)
    assert ref_vit and not new_vit  # transformers' input-grad hooks drag the frozen vision part into autograd
    assert ref.keys() == new.keys() and len(new) > 0
    assert not any(part in n for n in new for part in ("visual", "vision_tower", "multi_modal_projector"))
    for n in ref:
        assert torch.allclose(ref[n], new[n], atol=1e-6, rtol=1e-5), n
    assert any(g.abs().sum() > 0 for g in new.values())


def test_lora_targets_only_the_language_model():
    for name in ("model.language_model.layers.0.self_attn.q_proj", "model.layers.3.mlp.down_proj",
                 "model.language_model.layers.27.mlp.gate_proj"):
        assert re.match(LORA_TARGET_REGEX, name), name
    for name in ("model.visual.blocks.0.attn.qkv", "model.vision_tower.encoder.layer.0.attention.q_proj",
                 "model.multi_modal_projector.linear_1"):
        assert not re.match(LORA_TARGET_REGEX, name), name
