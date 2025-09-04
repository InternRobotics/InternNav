import numpy as np
import pytest

from internnav.evaluator.vln_pe_evaluator import transform_action_batch


@pytest.mark.slow
def test_transform_action_batch_discrete():
    origin = [(np.array([0]),), (np.array([-1]),), (np.array([3]),)]
    out = transform_action_batch(origin, flash=False)
    assert out == [
        {'h1': {'stop': []}},
        {'h1': {'stand_still': []}},
        {'h1': {'move_by_discrete': [3]}},
    ]


@pytest.mark.slow
def test_transform_action_batch_flash():
    origin = [(np.array([5]),)]
    out = transform_action_batch(origin, flash=True)
    assert out == [{'h1': {'move_by_flash': [5]}}]
