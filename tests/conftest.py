import pytest


@pytest.fixture
def tmp_cfg(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("hello: world\n")
    return p


# 按需全局 hook（例：缺 GPU 时自动跳过 gpu 标记）
def pytest_runtest_setup(item):
    if "gpu" in item.keywords:
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("No CUDA for gpu-marked test")
        except Exception:
            pytest.skip("Torch not available")
    if "ray" in item.keywords:
        try:
            import ray

            ray.init()
            assert ray.is_initialized()
        except Exception:
            pytest.skip("ray not available")
