import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


if importlib.util.find_spec("torch") is None:
    sys.modules["torch"] = SimpleNamespace(is_tensor=lambda _value: False)


def _load_module():
    path = Path(__file__).parents[2] / "internnav" / "evaluator" / "utils" / "diagnostic_logger.py"
    spec = importlib.util.spec_from_file_location("diagnostic_logger", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnostic_logger"] = module
    spec.loader.exec_module(module)
    return module


diagnostic_logger = _load_module()


def test_project_pixel_point_scales_model_coordinates_to_video_frame():
    assert diagnostic_logger.project_pixel_point((192, 192), (384, 384), (640, 480)) == (320, 240)


def test_save_s2_decision_draws_goal_and_samples_depth(tmp_path):
    logger = diagnostic_logger.DiagnosticLogger(str(tmp_path), {"run_id": "test"})
    logger.start_episode({"scene_id": "scene", "episode_id": 1})
    image = Image.new("RGB", (384, 384), (20, 30, 40))
    depth = np.full((480, 640), 2.5, dtype=np.float32)

    result = logger.save_s2_decision(
        3,
        image,
        decision_step=7,
        output_type="pixel_goal",
        raw_output="192 192",
        pixel_goal=(192, 192),
        current_subtask="walk to the doorway",
        depth_m=depth,
        model_image_size=(384, 384),
        generation_confidence=0.82,
        confidence_threshold=0.35,
        confidence_gate="accepted",
    )

    output = tmp_path / result["decision_image"]
    raw_output = tmp_path / result["decision_raw_image"]
    assert output.is_file()
    assert raw_output.is_file()
    assert result["decision_raw_image_size"] == [384, 384]
    assert result["decision_image_size"] == [384, 384]
    assert result["decision_point"] == [192, 192]
    assert result["decision_display_point"] == [192, 192]
    assert result["decision_coordinate_image_size"] == [384, 384]
    assert abs(result["decision_point_depth_m"] - 2.5) < 1e-6
    assert result["decision_generation_confidence"] == 0.82
    assert result["decision_confidence_threshold"] == 0.35
    assert result["decision_confidence_gate"] == "accepted"
    assert np.asarray(Image.open(output))[192, 192].tolist() != [20, 30, 40]
    assert np.max(np.abs(np.asarray(Image.open(raw_output), dtype=np.int16) - [20, 30, 40])) <= 2


def test_save_s2_decision_scales_original_row_col_coordinates(tmp_path):
    logger = diagnostic_logger.DiagnosticLogger(str(tmp_path), {"run_id": "test"})
    logger.start_episode({"scene_id": "scene", "episode_id": 3})
    image = Image.new("RGB", (640, 480), (20, 30, 40))
    depth = np.full((480, 640), 1.75, dtype=np.float32)

    result = logger.save_s2_decision(
        2,
        image,
        decision_step=8,
        output_type="pixel_goal",
        raw_output="500 300",
        pixel_goal=(300, 500),
        depth_m=depth,
        model_image_size=(384, 384),
    )

    assert result["decision_point"] == [300, 500]
    assert result["decision_display_point"] == [300, 240]
    assert result["decision_coordinate_image_size"] == [640, 480]
    assert abs(result["decision_point_depth_m"] - 1.75) < 1e-6


def test_save_s2_decision_marks_stop_without_a_goal(tmp_path):
    logger = diagnostic_logger.DiagnosticLogger(str(tmp_path), {"run_id": "test"})
    logger.start_episode({"scene_id": "scene", "episode_id": 2})
    result = logger.save_s2_decision(
        1,
        Image.new("RGB", (384, 384), (0, 0, 0)),
        decision_step=4,
        output_type="stop",
        raw_output="STOP",
        action_names=["STOP"],
    )
    assert (tmp_path / result["decision_image"]).is_file()
    assert result["decision_point"] is None
    assert result["decision_action_names"] == ["STOP"]


def test_save_semantic_shadow_draws_separate_anchor_and_goal(tmp_path):
    logger = diagnostic_logger.DiagnosticLogger(str(tmp_path), {"run_id": "test"})
    logger.start_episode({"scene_id": "scene", "episode_id": 4})
    result = logger.save_semantic_shadow(
        5,
        Image.new("RGB", (384, 384), (20, 30, 40)),
        decision_step=9,
        anchor="stove",
        anchor_point=(500, 160),
        goal_point=(120, 220),
        uncertain=True,
        progress_step=1,
        stop_complete=False,
        reason="multiple appliances",
        raw_output="{}",
        generation_confidence=0.7,
    )
    output = tmp_path / result["shadow_image"]
    assert output.is_file()
    assert result["shadow_anchor_display_point"] == [300, 128]
    assert result["shadow_goal_display_point"] == [72, 176]
    assert result["shadow_coordinate_image_size"] == [640, 480]
