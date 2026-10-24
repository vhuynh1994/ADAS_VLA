"""Deployment form of Qwen2.5-VL (deploy/qwen_vl.py) against transformers on a tiny random model: image layout,
ViT with window masks, M-RoPE positions, LLM parts run like Genie (chunked prefill + one-token decode in a KV
cache), and the Hadamard rotation. Needs torch + transformers (skipped in the minimal CI environment)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from adas_vla.deploy import qwen_vl as qv  # noqa: E402

W, H = 168, 140  # 12 x 10 patches -> 6 x 5 merge units -> windows of 4x4 units: 4 windows of 16, 8, 4, 2 units


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen2_5_VLConfig(
        text_config={"hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "vocab_size": 300, "rms_norm_eps": 1e-6,
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 48, "num_heads": 2, "out_hidden_size": 64,
                       "fullatt_block_indexes": [1], "window_size": 112, "patch_size": 14, "spatial_merge_size": 2,
                       "temporal_patch_size": 2},
        image_token_id=290, vision_start_token_id=291, vision_end_token_id=292, tie_word_embeddings=True)
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    with torch.no_grad():  # random init leaves some weights at 0 / 1; make every parameter matter
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    path = tmp_path_factory.mktemp("tiny_qwen_vl")
    model.save_pretrained(path)
    spec = qv.Spec.from_model(path, image_w=W, image_h=H)
    return model, path, spec


def _vit(model, spec, pix):
    grid = torch.tensor([[1, spec.grid_h, spec.grid_w]])
    out = model.model.visual(pix, grid_thw=grid)
    return out.pooler_output if hasattr(out, "pooler_output") else out


def test_patchify_matches_image_processor(tiny):
    from PIL import Image
    from transformers import Qwen2VLImageProcessor

    _, _, spec = tiny
    rgb = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    ref = Qwen2VLImageProcessor(min_pixels=28 * 28, max_pixels=W * H)(images=Image.fromarray(rgb), return_tensors="pt")
    assert ref["image_grid_thw"].tolist() == [[1, spec.grid_h, spec.grid_w]]
    np.testing.assert_allclose(qv.patchify(rgb, spec), ref["pixel_values"].numpy(), atol=2e-6)


@pytest.mark.parametrize("rotate_vit", [False, True])
def test_vision_encoder_matches_transformers(tiny, rotate_vit):
    model, path, spec = tiny
    pix = torch.randn(spec.patches, spec.patch_dim)
    vit_rot = qv.orthogonal(spec.vit_dim, seed=5) if rotate_vit else None
    vit = qv.VisionEncoder(qv.Checkpoint(path), spec, vit_rotation=vit_rot).eval()
    inputs = qv.vit_inputs(spec)
    with torch.no_grad():
        mine = vit(pix, *(inputs[k] for k in qv.VisionEncoder.INPUTS[1:]))
        ref = _vit(model, spec, pix)
    assert mine.shape == (spec.image_tokens, 64)
    torch.testing.assert_close(mine, ref, atol=1e-4, rtol=1e-4)
    order, window = qv.vit_layout(spec)
    assert sorted(order.tolist()) == list(range(spec.image_tokens))
    assert np.bincount(window.numpy()).tolist() == [64, 32, 16, 8]  # patches per window (4 per merge unit)


def _prompt(spec, n_before=5, n_after=7):
    rng = np.random.default_rng(1)
    before, after = rng.integers(0, 280, n_before).tolist(), rng.integers(0, 280, n_after).tolist()
    ids = torch.tensor(before + [spec.image_token] * spec.image_tokens + after)
    return before, after, ids


def test_positions_match_get_rope_index(tiny):
    model, _, spec = tiny
    before, after, ids = _prompt(spec)
    ref, _ = model.model.get_rope_index(ids[None], (ids == spec.image_token).long()[None],
                                        torch.tensor([[1, spec.grid_h, spec.grid_w]]))
    assert torch.equal(qv.prompt_positions(len(before), len(after), spec), ref[:, 0])


@pytest.mark.parametrize("rotate", [False, True, "heads"])
def test_parts_with_kv_cache_match_transformers(tiny, rotate):
    model, path, spec = tiny
    ck = qv.Checkpoint(path)
    rot = qv.hadamard(spec.dim, seed=3) if rotate else None
    head_rot = qv.hadamard(spec.head_dim, seed=4) if rotate == "heads" else None
    before, after, ids = _prompt(spec)
    pix = torch.randn(spec.patches, spec.patch_dim, generator=torch.Generator().manual_seed(6))
    grid = torch.tensor([[1, spec.grid_h, spec.grid_w]])
    with torch.no_grad():
        ref = model(input_ids=ids[None], pixel_values=pix, image_grid_thw=grid,
                    mm_token_type_ids=(ids == spec.image_token).long()[None]).logits[0]
        vit = qv.VisionEncoder(ck, spec, rotation=rot).eval()
        inputs = qv.vit_inputs(spec)
        feats = vit(pix, *(inputs[k] for k in qv.VisionEncoder.INPUTS[1:]))
        lut = qv.embedding_table(ck, rot)
        parts = [qv.DecoderPart(ck, spec, r, last=i == len(qv.part_ranges(spec, 2)) - 1, rotation=rot,
                                head_rotation=head_rot).eval() for i, r in enumerate(qv.part_ranges(spec, 2))]
        gen = qv.Generator(parts, spec, cl=64, ar=8)
        embeds = torch.cat([lut[before], feats, lut[after]])
        last = gen.prefill(embeds, qv.prompt_positions(len(before), len(after), spec))
        torch.testing.assert_close(last, ref[-1], atol=2e-4, rtol=2e-4)
        # one decode step at the next position matches the full forward of the extended sequence
        nxt = int(last[:280].argmax())  # a text token (not the image / vision ids above 280)
        step = gen.decode(lut[nxt])
        ids2 = torch.cat([ids, torch.tensor([nxt])])
        ref2 = model(input_ids=ids2[None], pixel_values=pix, image_grid_thw=grid,
                     mm_token_type_ids=(ids2 == spec.image_token).long()[None]).logits[0, -1]
        torch.testing.assert_close(step, ref2, atol=2e-4, rtol=2e-4)


def test_part_io_names(tiny):
    _, path, spec = tiny
    parts = [qv.DecoderPart(qv.Checkpoint(path), spec, r, last=i == 1) for i, r in enumerate(qv.part_ranges(spec, 2))]
    ins, outs = parts[0].io_names()
    assert ins[:4] == ["inputs_embeds", "attention_mask", "position_ids_cos", "position_ids_sin"]
    assert outs[0] == "hidden_2" and parts[1].io_names()[0][0] == "hidden_2"
    assert parts[1].io_names()[1][:3] == ["logits", "past_key_2_out", "past_value_2_out"]


def test_hadamard_is_orthogonal():
    r = qv.hadamard(64, seed=1)
    torch.testing.assert_close(r @ r.T, torch.eye(64, dtype=torch.float64))
    q = qv.orthogonal(1280, seed=2)  # 256 x 5: Hadamard (x) random orthogonal
    torch.testing.assert_close(q @ q.T, torch.eye(1280, dtype=torch.float64))
    assert float(q.abs().max()) < 0.07  # one outlier channel spreads over all 1280 (identity: 1.0)


@pytest.mark.parametrize("block", [None, 16])
def test_gptq_on_grid_and_better_than_rounding(block):
    from adas_vla.deploy import gptq as G

    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, 64, generator=g) @ torch.randn(64, 64, generator=g)  # correlated inputs
    w = torch.randn(32, 64, generator=g)
    h = x.T @ x
    s = G.grid_scales(w, 4, block)
    q = G.gptq(w, h, s, 4)
    assert G.check_grid(q, s, 4) < 1e-4
    rtn = (w / s).round().clamp(-8, 7) * s
    out_err = lambda a: float(((x @ (w - a).T) ** 2).sum())  # noqa: E731
    assert out_err(q) < 0.8 * out_err(rtn)
    if block:  # LPBQ: every block scale is an integer multiple (1..16) of the row's scale
        sc, ints = G.lpbq_int_scales(s, block)
        torch.testing.assert_close(s[:, ::block], sc[:, None] * ints)
        assert int(ints.min()) >= 1 and int(ints.max()) <= 16
