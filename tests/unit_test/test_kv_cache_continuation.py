"""Opt-in CLI and tiny-model cache regressions; no checkpoints or simulator.

See kv_cache_continuation.md for Habitat measurements and their limitations.
"""

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    IMAGE_TOKEN_INDEX,
    InternVLAN1ForCausalLM,
    InternVLAN1ModelConfig,
)


@pytest.fixture
def entrypoint(monkeypatch):
    # Only replace the simulator entry point; exercise the real argument parser
    # and the repository's real configuration loading and override logic.
    evaluator = ModuleType('internnav.evaluator')
    evaluator.Evaluator = SimpleNamespace(init=Mock(return_value=SimpleNamespace(eval=Mock())))
    monkeypatch.setitem(sys.modules, 'internnav.evaluator', evaluator)
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('eval_cli_test', root / 'scripts/eval/eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, root


@pytest.mark.parametrize('enabled', [False, True])
def test_flag_only_changes_requested_setting(entrypoint, monkeypatch, enabled):
    module, root = entrypoint
    config = str(root / 'scripts/eval/configs/habitat_dual_system_cfg.py')
    original = module.load_eval_cfg(config)
    args = ['eval.py', '--config', config]
    if enabled:
        args.append('--kv-cache-continuation')
    monkeypatch.setattr(sys, 'argv', args)
    assert module.parse_args().kv_cache_continuation is enabled
    module.main()
    actual = module.Evaluator.init.call_args[0][0]
    if enabled:
        original.agent.model_settings['kv_cache_continuation'] = True
    assert actual == original
    module.Evaluator.init.return_value.eval.assert_called_once_with()


def test_rejects_flag_for_system2_only(entrypoint, monkeypatch):
    module, root = entrypoint
    config = str(root / 'scripts/eval/configs/habitat_s2_cfg.py')
    monkeypatch.setattr(sys, 'argv', ['eval.py', '--config', config, '--kv-cache-continuation'])
    with pytest.raises(ValueError, match='Habitat dual_system'):
        module.main()
    module.Evaluator.init.assert_not_called()


@pytest.fixture(scope='module', params=['eager', 'sdpa'])
def model(request):
    config = InternVLAN1ModelConfig(
        vocab_size=151680,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_scaling={'type': 'mrope', 'mrope_section': [2, 2, 2]},
        image_token_id=IMAGE_TOKEN_INDEX,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        eos_token_id=151645,
        pad_token_id=151643,
        n_query=4,
        system1='navdp',  # Synchronous NavDP constructs only the latent queries.
        vision_config={
            'depth': 1,
            'hidden_size': 24,
            'intermediate_size': 48,
            'num_heads': 2,
            'out_hidden_size': 24,
            'patch_size': 14,
            'spatial_merge_size': 2,
            'temporal_patch_size': 2,
            'fullatt_block_indexes': [0],
        },
    )
    config._attn_implementation = request.param
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            yield InternVLAN1ForCausalLM(config).eval()
    finally:
        torch.set_num_threads(previous_threads)


def make_inputs(image_count):
    grids = torch.tensor([[1, 4, 4], [1, 4, 6]][:image_count])
    ids = [30, 31]
    for grid in grids:
        ids += [151652] + [IMAGE_TOKEN_INDEX] * (int(grid.prod()) // 4) + [151653, 32]
    input_ids = torch.tensor([ids + [33, 34]])
    return {
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'pixel_values': torch.randn(int(grids.prod(dim=-1).sum()), 3 * 2 * 14 * 14),
        'image_grid_thw': grids,
    }


@torch.no_grad()
def generate(model, inputs, eos=False):
    return model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=3,
        eos_token_id=151645 if eos else None,
        forced_eos_token_id=151645 if eos else None,
        use_cache=True,
        return_dict_in_generate=True,
    )


@pytest.mark.parametrize('image_count', [1, 2])
@pytest.mark.parametrize('complete_cache', [False, True])
@pytest.mark.parametrize('eos', [False, True])
@torch.no_grad()
def test_matches_full_replay(model, image_count, complete_cache, eos):
    inputs = make_inputs(image_count)
    generated = generate(model, inputs, eos)
    ids, cache = generated.sequences, generated.past_key_values
    assert cache.get_seq_length() == ids.shape[1] - 1
    if eos:
        assert ids[0, -1] == 151645
    if complete_cache:
        positions, _ = model.get_rope_index(ids, inputs['image_grid_thw'])
        model(
            input_ids=ids[:, -1:],
            position_ids=positions[:, :, -1:],
            past_key_values=cache,
            cache_position=torch.tensor([ids.shape[1] - 1]),
            use_cache=True,
        )
    original_length = cache.get_seq_length()
    original_kv = [(key.clone(), value.clone()) for key, value in cache]
    full = model.generate_latents(ids, inputs['pixel_values'], inputs['image_grid_thw'])
    # An unrelated request may change this state before readout.
    model.rope_deltas = torch.tensor([[12345]])
    for _ in range(2):
        actual = model.generate_latents_from_cache(generated, inputs['image_grid_thw'], inputs['attention_mask'])
        torch.testing.assert_close(actual, full, atol=2e-6, rtol=2e-5)
        assert not actual.requires_grad
        assert cache.get_seq_length() == original_length
        for (key, value), (expected_key, expected_value) in zip(cache, original_kv):
            torch.testing.assert_close(key, expected_key, atol=0, rtol=0)
            torch.testing.assert_close(value, expected_value, atol=0, rtol=0)


@torch.no_grad()
def test_rejects_unsupported_inputs(model):
    inputs = make_inputs(1)
    generated = generate(model, inputs)
    grid = inputs['image_grid_thw']
    with pytest.raises(ValueError, match='DynamicCache'):
        model.generate_latents_from_cache(SimpleNamespace(sequences=generated.sequences), grid)
    with pytest.raises(ValueError, match='unpadded'):
        model.generate_latents_from_cache(generated, grid, torch.tensor([[0, 1]]))
    with pytest.raises(ValueError, match='unpadded'):
        model.generate_latents_from_cache(generated, grid, torch.ones(1, 1, 1, 1))
    with pytest.raises(ValueError, match='num_beams'):
        model.generate_latents_from_cache(SimpleNamespace(**dict(generated), beam_indices=torch.tensor([0])), grid)
    with pytest.raises(ValueError, match='batch size 1'):
        model.generate_latents_from_cache(
            SimpleNamespace(sequences=generated.sequences.repeat(2, 1), past_key_values=generated.past_key_values),
            grid,
        )
    bad = copy.deepcopy(generated)
    bad.past_key_values.crop(0)
    with pytest.raises(ValueError, match='final token'):
        model.generate_latents_from_cache(bad, grid)
    bad = copy.deepcopy(generated)
    bad.past_key_values.crop(bad.past_key_values.get_seq_length() - 1)
    with pytest.raises(ValueError, match='final token'):
        model.generate_latents_from_cache(bad, grid)
    with pytest.raises(ValueError, match='final token'):
        model.generate_latents_from_cache(
            SimpleNamespace(sequences=generated.sequences[:, :-2], past_key_values=generated.past_key_values),
            grid,
        )
    bad = copy.deepcopy(generated)
    bad.sequences[0, -1] = IMAGE_TOKEN_INDEX
    with pytest.raises(ValueError, match='ordinary text'):
        model.generate_latents_from_cache(bad, grid)
    model.config.use_sliding_window = True
    try:
        with pytest.raises(ValueError, match='full attention'):
            model.generate_latents_from_cache(generated, grid)
    finally:
        model.config.use_sliding_window = False
    model.train()
    try:
        with pytest.raises(ValueError, match='model.eval'):
            model.generate_latents_from_cache(generated, grid)
    finally:
        model.eval()


@torch.no_grad()
def test_avoids_vision_and_logits_and_restores_cache_on_error(model):
    inputs = make_inputs(1)
    generated = generate(model, inputs)
    cache = generated.past_key_values
    original_length = cache.get_seq_length()

    def unexpected_forward(*args):
        raise AssertionError('Readout must not run vision or the vocabulary head')

    hooks = [module.register_forward_pre_hook(unexpected_forward) for module in (model.visual, model.lm_head)]
    try:
        model.generate_latents_from_cache(generated, inputs['image_grid_thw'])
    finally:
        for hook in hooks:
            hook.remove()

    def fail_after_forward(*args):
        assert cache.get_seq_length() > original_length
        raise RuntimeError('simulated failure after cache update')

    hook = model.model.register_forward_hook(fail_after_forward)
    try:
        with pytest.raises(RuntimeError, match='simulated failure'):
            model.generate_latents_from_cache(generated, inputs['image_grid_thw'])
    finally:
        hook.remove()
    assert cache.get_seq_length() == original_length
