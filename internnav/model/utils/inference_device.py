import torch


def resolve_inference_device(device, attn_implementation='auto'):
    device = torch.device(device)
    supported = ('auto', 'flash_attention_2', 'sdpa', 'eager')
    if attn_implementation not in supported:
        raise ValueError(f'Attention must be one of {supported}, got {attn_implementation!r}')
    if device.type == 'xpu':
        if attn_implementation == 'flash_attention_2':
            raise ValueError('External FlashAttention 2 is not supported on XPU; use sdpa or eager')
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            raise RuntimeError(
                'XPU was requested but is unavailable; check native PyTorch XPU wheels and Intel drivers'
            )
        if device.index is not None and device.index >= torch.xpu.device_count():
            raise RuntimeError(f'Requested XPU device {device} does not exist')
    if attn_implementation == 'auto':
        attn_implementation = 'flash_attention_2' if device.type == 'cuda' else 'sdpa'
    return device, attn_implementation
