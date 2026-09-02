"""Regression coverage for the NextDiT checkpointing hook."""

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from internnav.model.basemodel.internvla_n1.nextdit_crossattn_traj import (
    NextDiTCrossAttn,
    NextDiTCrossAttnConfig,
)


def test_nextdit_enables_and_disables_gradient_checkpointing():
    config = NextDiTCrossAttnConfig(
        input_size=4,
        patch_size=1,
        in_channels=4,
        dim=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        multiple_of=8,
        latent_embedding_size=16,
        _gradient_checkpointing=True,
    )

    model = NextDiTCrossAttn(config)
    assert model.model.gradient_checkpointing is True
    assert model.model.is_gradient_checkpointing is True

    model.model.disable_gradient_checkpointing()
    assert model.model.gradient_checkpointing is False
    assert model.model.is_gradient_checkpointing is False
