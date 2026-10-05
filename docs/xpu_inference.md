# Optional Intel XPU eager inference

Eager inference for InternVLA-N1-DualVLN on Intel GPUs.

## Installation

Install the Intel GPU driver and runtime per the
[PyTorch Intel GPU prerequisites](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).
Then, from the repository root, in a dedicated virtual environment:

```bash
git submodule update --init internnav/model/basemodel/LongCLIP
venv/bin/python -m pip install -r requirements/internvla_n1_xpu.txt
venv/bin/python -m pip check
venv/bin/python -c "import torch; assert torch.xpu.is_available(); print(torch.__version__); print(torch.xpu.get_device_properties(0)); print(torch.xpu.mem_get_info(0))"
```

Do not install `xformers` or `flash-attn` in this environment. Keep `diffusers==0.32.2`.

## Auxiliary checkpoint

Download the Depth Anything V2 checkpoint into `checkpoints/`:

```bash
curl -fL --retry 2 -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
sha256sum checkpoints/depth_anything_v2_metric_hypersim_vits.pth
```

## Configuration

Set `device` and, optionally, `attn_implementation` (default `auto`) in `model_settings`
of the evaluation config, e.g. `scripts/eval/configs/h1_internvla_n1_async_cfg.py`:

```python
model_settings={
    ...
    'model_path': 'checkpoints/InternVLA-N1-DualVLN',
    'device': 'xpu:0',
    'attn_implementation': 'auto',
}
```

| Device | `auto` | Other selections |
| --- | --- | --- |
| CUDA | `flash_attention_2` | `sdpa`, `eager` |
| XPU | `sdpa` | `eager`; `flash_attention_2` is not supported |

The real-world HTTP server uses `auto`.

## Launch

```bash
venv/bin/python -m scripts.realworld.http_internvla_server \
  --model_path checkpoints/InternVLA-N1-DualVLN \
  --device xpu:0 --plan_step_gap 4 --skip_warmup
```

The server listens on port 5801.

## Memory

Model weights alone need 15.62 GiB of free GPU memory; inference requires more.

## Tests

```bash
venv/bin/python -m pip install pytest
INTERNNAV_FULL_XPU_TEST=1 venv/bin/python -m pytest tests/unit_test/test_xpu_inference.py -q -s
```

CUDA equivalence tests, in a CUDA environment:

```bash
python -m pytest tests/unit_test/test_xpu_inference.py -q -rs -k "accelerator or cuda"
```