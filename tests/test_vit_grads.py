import copy

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
    return transformers.Qwen2_5_VLForConditionalGeneration(cfg)


def _lora_grads(model, input_hook) -> tuple[dict, bool]:
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(1)  # same LoRA init in both runs
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGET_REGEX,
                                                       init_lora_weights=False))
    input_hook(model)
    model.train()
    vit_grad = []  # do the image features leave the ViT with an autograd graph?
    model.base_model.model.model.visual.merger.register_forward_hook(
        lambda m, i, o: vit_grad.append(o.requires_grad))
    ids = torch.tensor([[5, 6, VSTART, IMAGE, IMAGE, IMAGE, IMAGE, VEND, 7, 8, 9, 10]])
    torch.manual_seed(2)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids,
                pixel_values=torch.randn(16, 3 * 2 * 14 * 14), image_grid_thw=torch.tensor([[1, 4, 4]]))
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    return grads, vit_grad[0]


def test_lora_gradients_without_backprop_through_the_frozen_vit():
    base = _tiny_qwen25_vl()
    ref, ref_vit = _lora_grads(copy.deepcopy(base), lambda m: None)
    new, new_vit = _lora_grads(copy.deepcopy(base), freeze_vision_autograd)
    assert ref_vit and not new_vit  # transformers' input-grad hooks drag the frozen ViT into autograd
    assert ref.keys() == new.keys() and len(new) > 0 and not any("visual" in n for n in new)
    for n in ref:
        assert torch.allclose(ref[n], new[n], atol=1e-6, rtol=1e-5), n
    assert any(g.abs().sum() > 0 for g in new.values())
