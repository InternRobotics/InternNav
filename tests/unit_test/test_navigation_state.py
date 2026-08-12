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


navigation_state = _load_module("navigation_state")
recovery_module = _load_module("recovery_controller")
trajectory_module = _load_module("trajectory_selector")

DepthObservationSummarizer = navigation_state.DepthObservationSummarizer
CameraPitchState = navigation_state.CameraPitchState
InstructionStateTracker = navigation_state.InstructionStateTracker
PixelGoalMemory = navigation_state.PixelGoalMemory
S2ConfidenceGate = navigation_state.S2ConfidenceGate
SemanticShadowMonitor = navigation_state.SemanticShadowMonitor
build_stage_probe_prompt = navigation_state.build_stage_probe_prompt
estimate_stage_progress = navigation_state.estimate_stage_progress
parse_stage_completion_output = navigation_state.parse_stage_completion_output
bound_actions_at_lookdown = navigation_state.bound_actions_at_lookdown
parse_semantic_shadow_output = navigation_state.parse_semantic_shadow_output
extract_reference_landmark = navigation_state.extract_reference_landmark
select_history_indices = navigation_state.select_history_indices
select_uniform_history_indices = navigation_state.select_uniform_history_indices
split_instruction = navigation_state.split_instruction
RecoveryController = recovery_module.RecoveryController
TrajectorySelector = trajectory_module.TrajectorySelector


def _constant_candidates(offsets, steps=4):
    candidates = np.zeros((len(offsets), steps, 3), dtype=np.float32)
    for index, lateral_delta in enumerate(offsets):
        candidates[index, :, 0] = 1.0
        candidates[index, :, 1] = lateral_delta
    return candidates


def test_instruction_split_exposes_turn_stair_and_stop_state():
    clauses = split_instruction("Turn right and step down two stairs and stop.")
    assert clauses == ("Turn right", "step down two stairs", "stop")


def test_stage_probe_keeps_landmark_modifier_and_all_ordered_clauses():
    instruction = (
        "Walk towards the hallway with a blue painting. "
        "Turn left and wait by the entrance of the empty room."
    )
    clauses = split_instruction(instruction)
    assert clauses == (
        "Walk towards the hallway with a blue painting",
        "Turn left",
        "wait by the entrance of the empty room",
    )
    prompt = build_stage_probe_prompt(instruction, clauses, 0, image_count=5)
    assert "hallway with a blue painting" in prompt
    assert "entrance of the empty room" in prompt
    assert "5 chronological images" in prompt


def test_stage_progress_catches_up_only_across_consecutive_complete_prefix():
    estimate = estimate_stage_progress(0, ["STOP", "STOP", "→"], clause_count=3)
    assert estimate.inferred_index == 2
    assert estimate.status == "INCOMPLETE"
    assert estimate.clause_results == ("COMPLETE", "COMPLETE", "INCOMPLETE")

    uncertain = estimate_stage_progress(0, ["STOP", "←", "STOP"], clause_count=3)
    assert uncertain.inferred_index == 1
    assert uncertain.status == "UNCERTAIN"
    assert uncertain.clause_results == ("COMPLETE", "UNCERTAIN")


def test_stage_probe_native_vocabulary_parser_is_conservative():
    assert parse_stage_completion_output("STOP") == "COMPLETE"
    assert parse_stage_completion_output("→→→→") == "INCOMPLETE"
    assert parse_stage_completion_output("←←←←") == "UNCERTAIN"
    assert parse_stage_completion_output("318 276") == "UNCERTAIN"


def test_two_step_descent_requires_measured_height_change():
    tracker = InstructionStateTracker("Step down two stairs and stop.", initial_height=2.0)
    assert tracker.stair_mode
    assert tracker.observe(1.9, [0.0], [0.0, 0.0]) is None
    assert tracker.observe(1.70, [0.0], [0.1, 0.0]) == "stair_complete"
    assert tracker.current_clause == "stop"


def test_top_of_stairs_requires_height_and_a_plateau():
    tracker = InstructionStateTracker("Walk up the stairs and stop at the top.", initial_height=0.0)
    for height in (0.2, 0.4, 0.65):
        assert tracker.observe(height, [0.0], [0.0, 0.0]) is None
    for height in (0.66, 0.65, 0.66):
        event = tracker.observe(height, [0.0], [0.0, 0.0])
    assert event == "stair_complete"
    assert tracker.terminal_stop_evidence()
    assert not tracker.should_force_stop()
    assert tracker.should_force_stop(stall_confirmed=True)


def test_very_top_stop_is_merged_into_the_preceding_stair_clause():
    clauses = split_instruction(
        "Turn left and walk into the small hallway and up the stairs. "
        "Stop at the very top of the stairs."
    )
    assert clauses == (
        "Turn left",
        "walk into the small hallway and up the stairs and Stop at the very top of the stairs",
    )


def test_s2_marker_cannot_override_missing_stair_height_evidence():
    tracker = InstructionStateTracker("Go down the stairs, then turn right.", initial_height=1.0)
    assert not tracker.mark_model_completion("SUBTASK_DONE (120, 80)")
    assert tracker.stair_mode


def test_unverified_stair_stop_requires_height_evidence_instead_of_retry_count():
    tracker = InstructionStateTracker("Go down the stairs and stop.", initial_height=1.0)
    assert tracker.should_reject_stop()
    assert tracker.should_reject_stop()
    assert tracker.should_reject_stop()
    assert tracker.observe(0.4, [0.0], [0.0, 0.0]) == "stair_complete"
    assert not tracker.should_reject_stop()


def test_stop_is_rejected_when_an_ordered_future_stair_clause_is_unresolved():
    tracker = InstructionStateTracker(
        "Head into the hall, then go down the stairs and stop.", initial_height=1.0
    )
    assert tracker.current_clause == "Head into the hall"
    assert tracker.should_reject_stop()
    assert tracker.stop_rejections == 1


def test_non_terminal_completion_does_not_force_stop():
    tracker = InstructionStateTracker("Turn left and stop.", initial_compass=0.0)
    assert tracker.observe(0.0, [np.deg2rad(70)], [0.0, 0.0]) == "turn_complete"
    assert not tracker.should_force_stop()


def test_low_head_depth_identifies_open_centre_and_side_barriers():
    depth = np.full((120, 180), 3.0, dtype=np.float32)
    depth[:, :60] = 0.35
    depth[:, 120:] = 0.40
    summary = DepthObservationSummarizer().summarize(depth, stair_mode=True)
    assert summary.centre_open
    assert not summary.centre_blocked
    assert "railings" in summary.text


def test_failed_duplicate_goal_is_rejected_at_retry_limit():
    memory = PixelGoalMemory(retry_limit=2)
    first = memory.evaluate([100, 80], [0.0, 0.0], [0.0], previous_goal_failed=False)
    retry = memory.evaluate([105, 83], [0.1, 0.0], [0.05], previous_goal_failed=True)
    rejected = memory.evaluate([103, 82], [0.1, 0.0], [0.04], previous_goal_failed=True)
    assert not first.duplicate
    assert retry.duplicate and not retry.reject
    assert rejected.duplicate and rejected.reject


def test_goal_from_a_changed_heading_is_not_a_duplicate():
    memory = PixelGoalMemory(retry_limit=1, heading_tolerance_deg=20)
    memory.evaluate([100, 80], [0.0, 0.0], [0.0])
    decision = memory.evaluate([100, 80], [0.0, 0.0], [np.deg2rad(45)], previous_goal_failed=True)
    assert not decision.duplicate
    assert not decision.reject


def test_history_selection_excludes_current_and_keeps_recent_frames():
    selected = select_history_indices(frame_count=20, num_history=6, prioritize_recent=True)
    assert len(selected) == 6
    assert 19 not in selected
    assert selected[-1] == 18
    assert sum(index >= 15 for index in selected) >= 4


def test_uniform_history_selection_restores_nonsemantic_sampling():
    selected = select_uniform_history_indices(frame_count=20, num_history=6)
    assert selected == [0, 3, 7, 10, 14, 18]
    assert 19 not in selected


def test_failed_route_sector_changes_candidate_selection_without_hard_ban():
    candidates = _constant_candidates([1.0] * 8 + [-1.0] * 8)
    selection = TrajectorySelector().select(candidates, failed_route_directions={"left": 2})
    assert selection.route_direction == "right"
    assert selection.failed_route_penalty == 0.0


def test_recovery_records_route_sector_and_chooses_opposite_observation_turn():
    controller = RecoveryController()
    controller.set_route_direction("left")
    controller.update(1, [0.0, 0.0], [0.0, 0.0], collision=True)
    assert controller.failed_route_directions["left"] == 1
    assert controller.exploratory_turn() == 3
    assert "left x1" in controller.failed_route_context()


def test_repeated_direct_turn_uses_depth_gated_forward_once():
    controller = RecoveryController(max_consecutive_turns=4, direct_turn_escape_clearance=0.55)
    gps = np.asarray([0.0, 0.0], dtype=np.float32)
    for _ in range(4):
        decision = controller.update(3, gps, gps, is_s1=False)
    assert decision.reason == "repeated_turns"
    assert controller.consume_turn_stall()
    assert not controller.consume_turn_stall()

    actions, reason = controller.filter_direct_actions(
        [3, 3, 3, 3], centre_clearance=0.9, centre_blocked=False
    )
    assert actions == [1]
    assert reason == "depth_gated_forward_after_repeated_direct_turn"
    assert controller.filter_direct_actions([3, 3], 0.9, False) == ([3, 3], None)


def test_repeated_direct_turn_uses_opposite_probe_when_blocked():
    controller = RecoveryController(max_consecutive_turns=2)
    gps = np.asarray([0.0, 0.0], dtype=np.float32)
    controller.update(2, gps, gps, is_s1=False)
    controller.update(2, gps, gps, is_s1=False)

    actions, reason = controller.filter_direct_actions(
        [2, 2, 2], centre_clearance=0.3, centre_blocked=True
    )
    assert actions == [3]
    assert reason == "opposite_probe_after_repeated_direct_turn"


def test_temporary_low_head_view_requires_matching_restore_steps():
    camera = CameraPitchState()
    camera.record_look_down(count=2)
    assert camera.restore_steps == 2
    camera.record_look_up()
    assert camera.restore_steps == 1
    camera.mark_horizontal()
    assert camera.restore_steps == 0


def test_lookdown_ends_the_direct_action_queue():
    assert bound_actions_at_lookdown([3, 3, 5, 1, 1]) == [3, 3, 5]
    assert bound_actions_at_lookdown([5, 5, 5]) == [5]
    assert bound_actions_at_lookdown([1, 2, 3]) == [1, 2, 3]


def test_low_confidence_pixel_goal_requires_observation_turn():
    gate = S2ConfidenceGate(pixel_goal_threshold=0.35)
    assert gate.evaluate_pixel_goal(0.34).reject
    assert not gate.evaluate_pixel_goal(0.35).reject
    assert not gate.evaluate_pixel_goal(None).reject


def test_low_confidence_stop_requires_two_consecutive_decisions():
    gate = S2ConfidenceGate(stop_threshold=0.75)
    first = gate.evaluate_stop(0.40)
    second = gate.evaluate_stop(0.45)
    assert first.reject and not first.confirmed
    assert not second.reject and second.confirmed


def test_non_stop_breaks_low_confidence_stop_confirmation():
    gate = S2ConfidenceGate(stop_threshold=0.75)
    assert gate.evaluate_stop(0.40).reject
    gate.observe_non_stop()
    assert gate.evaluate_stop(0.40).reject
    assert not gate.evaluate_stop(0.80).reject


def test_semantic_shadow_signal_is_diagnostic_and_rate_limited():
    monitor = SemanticShadowMonitor(
        generation_threshold=0.55,
        minimum_token_threshold=0.20,
        query_interval=3,
        max_queries=2,
    )
    first = monitor.assess(0, "go through the first door", "120 200", 0.8, 0.5)
    assert first.query and first.semantic_risk
    assert not first.uncertainty_signal
    second = monitor.assess(1, "go through the first door", "120 200", 0.4, 0.1)
    assert not second.query
    assert second.uncertainty_signal
    assert set(second.reasons) == {
        "low_generation_confidence",
        "low_minimum_token_confidence",
    }
    third = monitor.assess(3, "go through the first door", "STOP", 0.8, 0.5)
    assert third.query


def test_parse_semantic_shadow_json_keeps_xy_coordinate_order():
    parsed = parse_semantic_shadow_output(
        '{"uncertain": true, "anchor": "stove", "anchor_point": [500, 160], '
        '"goal_point": [120, 220], "progress_step": 2, "stop_complete": false, '
        '"reason": "multiple appliances"}'
    )
    assert parsed["uncertain"] is True
    assert parsed["anchor_point"] == [500, 160]
    assert parsed["goal_point"] == [120, 220]
    assert parsed["progress_step"] == 2
    assert parsed["stop_complete"] is False


def test_extract_reference_landmark_prefers_relation_anchor():
    assert extract_reference_landmark(
        "Walk through the entry way to the left of the stove"
    ) == "stove"
    assert extract_reference_landmark(
        "Walk through the arched entry way that leads into the tiled room"
    ) == "arched entry way"
