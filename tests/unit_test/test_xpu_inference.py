from unittest.mock import patch

import pytest
import torch

from internnav.model.utils.inference_device import resolve_inference_device
from internnav.model.encoder.depth_anything.depth_anything_v2.dinov2_layers import attention as dino_attention


def _require_accelerator(device_type):
    backend = getattr(torch, device_type, None)
    if backend is None or not backend.is_available():
        pytest.skip(f'{device_type} unavailable')
    return backend


def _require_supported_transformers():
    import transformers
    from packaging.version import Version

    # Transformers 4.52 restructured Qwen2.5-VL modules that InternVLA-N1 builds on.
    if Version(transformers.__version__).release[:2] != (4, 51):
        pytest.skip(f'InternVLA-N1 requires transformers==4.51.0 (installed {transformers.__version__})')


def test_cuda_default_attention():
    device, attention = resolve_inference_device('cuda:0')
    assert str(device) == 'cuda:0'
    assert attention == 'flash_attention_2'


@pytest.mark.parametrize('attention', ['auto', 'sdpa', 'eager'])
def test_xpu_attention(attention):
    with patch.object(torch.xpu, 'is_available', return_value=True), patch.object(
        torch.xpu, 'device_count', return_value=1
    ):
        device, resolved = resolve_inference_device('xpu:0', attention)
    assert str(device) == 'xpu:0'
    assert resolved == ('sdpa' if attention == 'auto' else attention)


def test_xpu_rejects_flash_attention():
    with pytest.raises(ValueError, match='FlashAttention 2'):
        resolve_inference_device('xpu:0', 'flash_attention_2')


def test_xpu_unavailable():
    with patch.object(torch.xpu, 'is_available', return_value=False):
        with pytest.raises(RuntimeError, match='unavailable'):
            resolve_inference_device('xpu:0')


def test_xpu_invalid_index():
    with patch.object(torch.xpu, 'is_available', return_value=True), patch.object(
        torch.xpu, 'device_count', return_value=1
    ):
        with pytest.raises(RuntimeError, match='does not exist'):
            resolve_inference_device('xpu:1')


def test_invalid_attention():
    with pytest.raises(ValueError, match='Attention must be'):
        resolve_inference_device('cpu', 'invalid')


@pytest.mark.parametrize('device', ['cpu', 'xpu:0'])
def test_dino_native_fallback(device):
    if device.startswith('xpu') and not torch.xpu.is_available():
        pytest.skip('Native XPU unavailable')
    layer = dino_attention.MemEffAttention(32, num_heads=4).to(device).eval()
    inputs = torch.randn(2, 8, 32, device=device)
    with patch.object(dino_attention, 'XFORMERS_AVAILABLE', False), patch.object(
        dino_attention, 'memory_efficient_attention', create=True, side_effect=AssertionError('Called xFormers')
    ), torch.no_grad():
        expected = dino_attention.Attention.forward(layer, inputs)
        torch.testing.assert_close(layer(inputs), expected)
        with pytest.raises(AssertionError, match='nested tensors'):
            layer(inputs, attn_bias=object())


def test_dino_cuda_xformers_matches_native():
    _require_accelerator('cuda')
    if not dino_attention.XFORMERS_AVAILABLE:
        pytest.skip('xFormers unavailable')
    torch.manual_seed(3)
    layer = dino_attention.MemEffAttention(32, num_heads=4).to('cuda:0').eval()
    inputs = torch.randn(2, 8, 32, device='cuda:0')
    with torch.no_grad():
        torch.testing.assert_close(layer(inputs), dino_attention.Attention.forward(layer, inputs), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('failure', ['missing_file', 'missing_weight', 'wrong_shape'])
def test_auxiliary_checkpoint_fails_closed(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace

    from internnav.model.basemodel.internvla_n1 import internvla_n1_arch as arch
    from internnav.model.encoder.depth_anything.depth_anything_v2 import dpt

    class TinyDepth(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.pretrained = torch.nn.Linear(2, 2)

    monkeypatch.setattr(dpt, 'DepthAnythingV2', TinyDepth)
    monkeypatch.setattr(arch, 'MODEL_PATH_TO', str(tmp_path))
    checkpoint = tmp_path / 'depth_anything_v2_metric_hypersim_vits.pth'
    if failure != 'missing_file':
        weights = TinyDepth().state_dict()
        if failure == 'missing_weight':
            del weights['pretrained.weight']
        else:
            weights['pretrained.weight'] = torch.zeros(3, 3)
        torch.save(weights, checkpoint)
    expected = FileNotFoundError if failure == 'missing_file' else RuntimeError
    with pytest.raises(expected):
        arch.build_depthanythingv2(SimpleNamespace())


def test_auxiliary_checkpoint_restores_weights(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from internnav.model.basemodel.internvla_n1 import internvla_n1_arch as arch
    from internnav.model.encoder.depth_anything.depth_anything_v2 import dpt

    class TinyDepth(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.pretrained = torch.nn.Linear(2, 2)

    weights = TinyDepth().state_dict()
    torch.save(weights, tmp_path / 'depth_anything_v2_metric_hypersim_vits.pth')
    monkeypatch.setattr(dpt, 'DepthAnythingV2', TinyDepth)
    monkeypatch.setattr(arch, 'MODEL_PATH_TO', str(tmp_path))
    rgb = arch.build_depthanythingv2(SimpleNamespace())
    torch.testing.assert_close(rgb.weight, weights['pretrained.weight'])
    torch.testing.assert_close(rgb.bias, weights['pretrained.bias'])


@pytest.mark.parametrize('device_type', ['xpu', 'cuda'])
@pytest.mark.parametrize('attention', ['auto', 'sdpa', 'eager'])
def test_accelerator_bf16_prefill_cache_and_latents(device_type, attention):
    import copy
    from pathlib import Path

    from PIL import Image
    from transformers import AutoProcessor

    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM,
        InternVLAN1ModelConfig,
    )

    backend = _require_accelerator(device_type)
    _require_supported_transformers()
    checkpoint = Path(__file__).resolve().parents[2] / 'checkpoints/InternVLA-N1-DualVLN'
    if not (checkpoint / 'preprocessor_config.json').exists():
        pytest.skip('Requires local DualVLN processor')
    device, attention = resolve_inference_device(f'{device_type}:0', attention)
    if attention == 'flash_attention_2':
        pytest.importorskip('flash_attn')
    config = InternVLAN1ModelConfig(
        vocab_size=152064, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        rope_scaling={'type': 'default', 'mrope_section': [1, 1, 2]},
        image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
        n_query=4,
        vision_config={
            'depth': 1, 'hidden_size': 32, 'intermediate_size': 64, 'num_heads': 4,
            'out_hidden_size': 32, 'patch_size': 14, 'spatial_merge_size': 2,
            'temporal_patch_size': 2, 'fullatt_block_indexes': [0],
        },
        attn_implementation='eager',
    )
    torch.manual_seed(7)
    reference = InternVLAN1ForCausalLM(config).eval().to(torch.bfloat16)
    reference.model.latent_queries = torch.nn.Parameter(torch.randn(1, 4, 32, dtype=torch.bfloat16))
    device_config = copy.deepcopy(config)
    device_config._attn_implementation = attention
    device_config.vision_config._attn_implementation = attention
    model = InternVLAN1ForCausalLM(device_config).eval().to(torch.bfloat16)
    model.model.latent_queries = torch.nn.Parameter(torch.empty(1, 4, 32, dtype=torch.bfloat16))
    model.load_state_dict(reference.state_dict())
    model.to(device)
    processor = AutoProcessor.from_pretrained(checkpoint, min_pixels=784, max_pixels=784, local_files_only=True)
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Stop at the door.'}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[Image.new('RGB', (28, 28), color='green')], return_tensors='pt')
    backend.reset_peak_memory_stats()
    with torch.no_grad():
        expected = reference(**inputs, use_cache=True).logits
        device_inputs = copy.deepcopy(inputs).to(device)
        actual = model(**device_inputs, use_cache=True)
        assert actual.past_key_values is not None
        torch.testing.assert_close(actual.logits.cpu(), expected, rtol=0.05, atol=0.01)
        output = model.generate(**device_inputs, max_new_tokens=2, min_new_tokens=2, do_sample=False, use_cache=True)
        assert output.shape[1] == inputs.input_ids.shape[1] + 2
        latent = model.generate_latents(output, device_inputs.pixel_values, device_inputs.image_grid_thw)
        assert latent.shape == (1, 4, 32)
        assert latent.device.type == device_type and latent.dtype == torch.bfloat16
        assert latent.isfinite().all()
    assert device_inputs.input_ids.dtype == torch.long
    assert device_inputs.image_grid_thw.dtype == torch.long
    print(f'{device_type} {attention} tiny Qwen peak allocated bytes: {backend.max_memory_allocated()}')


@pytest.mark.parametrize('device_type', ['xpu', 'cuda'])
def test_accelerator_bf16_trajectory_with_identical_noise(monkeypatch, device_type):
    import copy
    from pathlib import Path
    from types import SimpleNamespace

    from internnav.model.basemodel.internvla_n1 import internvla_n1 as model_module
    from internnav.model.basemodel.internvla_n1 import internvla_n1_arch as arch

    backend = _require_accelerator(device_type)
    checkpoint_dir = Path(__file__).resolve().parents[2] / 'checkpoints'
    if not (checkpoint_dir / 'depth_anything_v2_metric_hypersim_vits.pth').exists():
        pytest.skip('Requires auxiliary pretrained checkpoint')
    monkeypatch.setattr(arch, 'MODEL_PATH_TO', str(checkpoint_dir))

    class TrajectoryModel(torch.nn.Module):
        generate_traj = model_module.InternVLAN1ForCausalLM.generate_traj

        def __init__(self):
            super().__init__()
            self.cond_projector = torch.nn.Sequential(
                torch.nn.Linear(3584, 768), torch.nn.GELU(approximate='tanh'), torch.nn.Linear(768, 768)
            )
            self.rgb_model = arch.build_depthanythingv2(SimpleNamespace())
            self.memory_encoder = arch.MemoryEncoder()
            self.rgb_resampler = arch.QFormer()
            self.traj_dit, self.noise_scheduler = arch.build_traj_dit(SimpleNamespace())
            self.action_encoder = torch.nn.Linear(3, 384)
            self.pos_encoding = arch.SinusoidalPositionalEncoding(384)
            self.action_decoder = torch.nn.Linear(384, 3)
            for name, value in (('_resnet_mean', model_module._RESNET_MEAN), ('_resnet_std', model_module._RESNET_STD)):
                self.register_buffer(name, torch.tensor(value).view(1, 1, 3, 1, 1), persistent=False)

        def get_model(self):
            return self

        def get_system1_type(self):
            return 'nextdit_async'

    torch.manual_seed(11)
    reference = TrajectoryModel().eval().to(torch.bfloat16)
    model = copy.deepcopy(reference).to(f'{device_type}:0')
    noise = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    latent = torch.randn(1, 4, 3584, dtype=torch.bfloat16)
    images = torch.rand(1, 2, 224, 224, 3, dtype=torch.float64)

    def fixed_noise(shape, generator, device, dtype):
        assert tuple(shape) == tuple(noise.shape)
        return noise.clone().to(device=device, dtype=dtype)

    monkeypatch.setattr(model_module, 'randn_tensor', fixed_noise)
    backend.reset_peak_memory_stats()
    # xFormers has no CPU kernels, so the CPU reference must use the native DINO attention.
    with torch.no_grad(), patch.object(dino_attention, 'XFORMERS_AVAILABLE', False):
        expected = reference.generate_traj(latent, images, predict_step_nums=4, num_inference_steps=2, num_sample_trajs=2)
    with torch.no_grad():
        actual = model.generate_traj(
            latent.to(f'{device_type}:0'), images.to(f'{device_type}:0'),
            predict_step_nums=4, num_inference_steps=2, num_sample_trajs=2,
        )
    assert actual.device.type == device_type
    assert actual.isfinite().all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.05, atol=0.05)
    print(f'{device_type} trajectory peak allocated bytes: {backend.max_memory_allocated()}')


def test_traj_dit_shapes_match_checkpoint():
    import json
    from pathlib import Path

    from safetensors import safe_open

    from internnav.model.basemodel.internvla_n1.nextdit_crossattn_traj import NextDiTCrossAttn, NextDiTCrossAttnConfig

    checkpoint = Path(__file__).resolve().parents[2] / 'checkpoints/InternVLA-N1-DualVLN'
    index_path = checkpoint / 'model.safetensors.index.json'
    if not index_path.exists():
        pytest.skip('Local DualVLN checkpoint unavailable')
    prefix = 'model.traj_dit.'
    weight_map = {k: f for k, f in json.loads(index_path.read_text())['weight_map'].items() if k.startswith(prefix)}
    model_shapes = {
        prefix + k: tuple(v.shape) for k, v in NextDiTCrossAttn(NextDiTCrossAttnConfig(latent_embedding_size=768)).state_dict().items()
    }
    checkpoint_shapes = {}
    for key, file in weight_map.items():
        with safe_open(checkpoint / file, 'pt') as shard:
            checkpoint_shapes[key] = tuple(shard.get_slice(key).get_shape())
    assert model_shapes == checkpoint_shapes


def test_diffusers_checkpointing_toggle_compatibility():
    from internnav.model.basemodel.internvla_n1.nextdit_crossattn_traj import NextDiTCrossAttn, NextDiTCrossAttnConfig

    model = NextDiTCrossAttn(NextDiTCrossAttnConfig(n_layers=1)).model
    assert model.gradient_checkpointing
    model.disable_gradient_checkpointing()
    assert not model.gradient_checkpointing
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing
    model._set_gradient_checkpointing(model, value=False)
    assert not model.gradient_checkpointing


def test_realworld_history_reset_and_thread_inference(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    import numpy as np
    from transformers import BatchFeature

    from internnav.agent import internvla_n1_agent_realworld as agent_module

    class StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(('generate', torch.is_grad_enabled()))
            assert kwargs['input_ids'].dtype == torch.long
            return SimpleNamespace(sequences=torch.ones(1, 3, dtype=torch.long), past_key_values=None)

        def generate_latents(self, *args):
            self.calls.append(('latents', torch.is_grad_enabled()))
            return torch.zeros(1, 4, 3584, dtype=torch.bfloat16)

        def generate_traj(self, *args):
            self.calls.append(('trajectory', torch.is_grad_enabled()))
            return torch.zeros(2, 4, 3, dtype=torch.bfloat16)

    class StubProcessor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: self.response)
            self.response = 'STOP'
            self.image_counts = []

        def apply_chat_template(self, *args, **kwargs):
            return 'prompt'

        def __call__(self, text, images, return_tensors):
            self.image_counts.append(len(images))
            return BatchFeature({
                'input_ids': torch.ones(1, 2, dtype=torch.long),
                'pixel_values': torch.zeros(1, 3),
                'image_grid_thw': torch.ones(1, 3, dtype=torch.long),
            })

    model, processor = StubModel(), StubProcessor()
    monkeypatch.setattr(agent_module.InternVLAN1ForCausalLM, 'from_pretrained', lambda *args, **kwargs: model)
    monkeypatch.setattr(agent_module.AutoProcessor, 'from_pretrained', lambda *args, **kwargs: processor)
    monkeypatch.chdir(tmp_path)
    agent = agent_module.InternVLAN1AsyncAgent(SimpleNamespace(
        device='cpu', model_path='stub', attn_implementation='eager',
        resize_w=28, resize_h=28, num_history=2, plan_step_gap=4,
    ))
    agent.reset()
    arguments = (np.zeros((28, 28, 3), dtype=np.uint8), np.zeros((28, 28), dtype=np.float32), np.eye(4), 'Stop.', np.eye(4))
    with ThreadPoolExecutor(max_workers=1) as worker:
        result = worker.submit(agent.step, *arguments).result()
        assert result.output_action == [0]
        processor.response = '(10, 20)'
        result = worker.submit(agent.step, *arguments).result()
        assert result.output_pixel == [20, 10]
        assert result.output_trajectory is not None
        assert processor.image_counts == [1, 2]
        processor.response = 'STOP'
        result = worker.submit(agent.step, *arguments, look_down=True).result()
        assert result.output_action == [0]
        assert processor.image_counts == [1, 2, 3]
        assert agent.episode_idx == 2
        assert len(agent.conversation_history) == 3
    assert {name for name, enabled in model.calls} == {'generate', 'latents', 'trajectory'}
    assert not any(enabled for name, enabled in model.calls if name != 'trajectory')
    assert torch.is_grad_enabled()
    agent.reset()
    assert agent.episode_idx == 0 and agent.rgb_list == [] and agent.conversation_history == []
    assert agent.output_action is None and agent.output_latent is None and agent.past_key_values is None


def test_policy_async_worker_publishes_output(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from internnav.agent import internvla_n1_agent as agent_module
    from internnav.model.utils.vln_utils import S2Input, S2Output

    agent = agent_module.InternVLAN1Agent.__new__(agent_module.InternVLAN1Agent)
    agent.s2_input = S2Input(idx=3, should_infer=True)
    agent.s2_output = S2Output()
    agent.camera_intrinsic = None
    agent.s2_input_lock = threading.Lock()
    agent.s2_output_lock = threading.Lock()
    agent.s2_agent_lock = threading.Lock()
    agent.policy = SimpleNamespace(s2_step=lambda *args: S2Output(output_action=[0]))
    targets = []

    def stop_after_iteration(seconds):
        raise StopIteration('Worker iteration complete')

    monkeypatch.setattr(agent_module, 'time', SimpleNamespace(sleep=stop_after_iteration))
    monkeypatch.setattr(agent_module, 'threading', SimpleNamespace(
        Thread=lambda target: SimpleNamespace(start=lambda: targets.append(target))
    ))
    agent._start_s2_thread()
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(StopIteration, match='Worker iteration complete'):
            worker.submit(targets[0]).result()
    assert agent.s2_output.output_action == [0]
    assert agent.s2_output.idx == 3
    assert not agent.s2_input.should_infer and not agent.s2_output.is_infering


@pytest.mark.slow
def test_full_checkpoint_xpu_eager(tmp_path, monkeypatch):
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

    if os.environ.get('INTERNNAV_FULL_XPU_TEST') != '1':
        pytest.skip('Set INTERNNAV_FULL_XPU_TEST=1 for full checkpoint validation')
    if not torch.xpu.is_available():
        pytest.skip('Native XPU unavailable')
    _require_supported_transformers()
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / 'checkpoints/InternVLA-N1-DualVLN'
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    weight_bytes = index['metadata']['total_size']
    free_bytes, total_bytes = torch.xpu.mem_get_info()
    if weight_bytes + 2 * 1024**3 > free_bytes:
        pytest.skip(f'Full checkpoint needs {weight_bytes} weight bytes plus inference headroom; XPU free={free_bytes}, total={total_bytes}')
    from internnav.model.basemodel.internvla_n1 import internvla_n1_arch as arch

    monkeypatch.setattr(arch, 'MODEL_PATH_TO', str(root / 'checkpoints'))
    monkeypatch.chdir(tmp_path)
    torch.xpu.reset_peak_memory_stats()
    agent = InternVLAN1AsyncAgent(SimpleNamespace(
        device='xpu:0', attn_implementation='auto', model_path=str(checkpoint),
        resize_w=384, resize_h=384, num_history=2, plan_step_gap=4,
    ))
    agent.reset()
    image = Image.new('RGB', (384, 384), color='green')
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Stop at the door.'}]}]
    text = agent.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = agent.processor(text=[text], images=[image], return_tensors='pt').to(agent.device)

    @torch.no_grad()
    def infer():
        outputs = agent.model.generate(**inputs, max_new_tokens=4, min_new_tokens=2, do_sample=False, use_cache=True, return_dict_in_generate=True)
        assert outputs.past_key_values is not None
        latent = agent.model.generate_latents(outputs.sequences, inputs.pixel_values, inputs.image_grid_thw)
        trajectory = agent.step_s1(latent, torch.zeros(1, 2, 224, 224, 3, device=agent.device), None)
        assert trajectory.shape == (32, 32, 3) and trajectory.isfinite().all()
        arguments = (np.asarray(image), np.zeros((384, 384), dtype=np.float32), np.eye(4), 'Stop at the door.', np.eye(4))
        agent.step_s2(*arguments)
        agent.step_s2(*arguments)
        assert agent.episode_idx == 2
        agent.reset()
        assert agent.episode_idx == 0 and not agent.rgb_list

    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(infer).result()
    torch.xpu.synchronize()
    print(f'Full checkpoint XPU peak allocated bytes: {torch.xpu.max_memory_allocated()}')