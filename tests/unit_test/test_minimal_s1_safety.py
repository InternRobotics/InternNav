import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module(name):
    path = Path(__file__).parents[2] / "internnav" / "habitat_extensions" / "vln" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RecoveryController = _load_module("recovery_controller").RecoveryController
TrajectorySelector = _load_module("trajectory_selector").TrajectorySelector


def _constant_candidates(offsets, steps=4):
    candidates = np.zeros((len(offsets), steps, 3), dtype=np.float32)
    for index, lateral_delta in enumerate(offsets):
        candidates[index, :, 0] = 1.0
        candidates[index, :, 1] = lateral_delta
    return candidates


def test_medoid_is_an_actual_candidate_not_the_mean_between_modes():
    actions = _constant_candidates([-1.0, -1.0, 1.0, 1.0])
    selection = TrajectorySelector().select(actions)
    reconstructed = TrajectorySelector.reconstruct(actions)
    assert any(np.array_equal(selection.trajectory, candidate) for candidate in reconstructed)
    assert not np.allclose(selection.trajectory[:, 1], 0.0)


def test_adaptive_chunks_only_shorten_for_risk_or_recent_failure():
    selector = TrajectorySelector()
    consistent = _constant_candidates([0.0] * 32)
    bimodal = _constant_candidates([-1.0] * 16 + [1.0] * 16)
    assert selector.select(consistent).chunk_size == 4
    assert selector.select(consistent, recent_failure=True).chunk_size == 1
    selection = selector.select(bimodal)
    assert selection.bimodal
    assert selection.cluster_sizes == (16, 16)
    assert selection.chunk_size == 1


def test_depth_short_range_check_marks_forward_path_unsafe():
    actions = _constant_candidates([0.0] * 8)
    depth = np.full((120, 160), 0.60, dtype=np.float32)
    selection = TrajectorySelector().select(actions, depth=depth)
    assert selection.depth_risk
    assert selection.clearance < 0.12
    assert selection.chunk_size == 1


def test_safe_depth_keeps_four_step_chunk():
    actions = _constant_candidates([0.0] * 8)
    depth = np.full((120, 160), 5.0, dtype=np.float32)
    selection = TrajectorySelector().select(actions, depth=depth)
    assert not selection.depth_risk
    assert selection.chunk_size == 4


def test_collision_and_low_forward_displacement_cancel_s1_queue():
    controller = RecoveryController(min_forward_displacement=0.03, system2_retry_limit=3)
    collision = controller.update(1, [0, 0], [0.2, 0], collision=True)
    blocked = controller.update(1, [0, 0], [0.01, 0], collision=False)
    assert collision.cancel_remaining and collision.reason == "collision"
    assert blocked.cancel_remaining and blocked.reason == "low_gps_displacement"
    assert controller.retry_count == 2
    assert controller.failed_directions[1] == 2
    assert not blocked.replan_system2


def test_third_goal_failure_escalates_to_system2_and_turn_does_not_reset_it():
    controller = RecoveryController(system2_retry_limit=3)
    first = controller.update(1, [0, 0], [0, 0], collision=True)
    controller.update(2, [0, 0], [0, 0])
    second = controller.update(1, [0, 0], [0, 0], collision=True)
    third = controller.update(1, [0, 0], [0, 0], collision=True)
    assert not first.replan_system2
    assert not second.replan_system2
    assert third.replan_system2
    assert third.goal_retry_count == 3


def test_turns_and_non_s1_actions_do_not_trigger_low_displacement():
    controller = RecoveryController()
    assert not controller.update(2, [0, 0], [0, 0], is_s1=True).cancel_remaining
    assert not controller.update(1, [0, 0], [0, 0], collision=True, is_s1=False).cancel_remaining


def test_success_resets_goal_failures_but_preserves_episode_retry_count():
    controller = RecoveryController()
    controller.update(1, [0, 0], [0, 0])
    controller.update(1, [0, 0], [0.2, 0])
    assert controller.consecutive_failures == 0
    assert controller.goal_retry_count == 0
    assert controller.retry_count == 1


def test_full_same_direction_rotation_replans_without_blocking_normal_turns():
    controller = RecoveryController(max_consecutive_turns=24)
    for _ in range(23):
        decision = controller.update(3, [0, 0], [0, 0], is_s1=False)
        assert not decision.cancel_remaining
    decision = controller.update(3, [0, 0], [0, 0], is_s1=False)
    assert decision.cancel_remaining
    assert decision.replan_system2
    assert decision.reason == "repeated_turns"
    assert decision.turn_count == 24


def test_direction_change_and_forward_motion_reset_turn_watchdog():
    controller = RecoveryController(max_consecutive_turns=24)
    for _ in range(16):
        assert not controller.update(3, [0, 0], [0, 0], is_s1=False).cancel_remaining
    assert not controller.update(2, [0, 0], [0, 0], is_s1=False).cancel_remaining
    for _ in range(16):
        assert not controller.update(2, [0, 0], [0, 0], is_s1=False).cancel_remaining
    assert not controller.update(1, [0, 0], [0.25, 0], is_s1=False).cancel_remaining
    assert controller.consecutive_turns == 0
