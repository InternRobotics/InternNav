"""Deterministic evaluation helpers and crash-safe VLN diagnostic logging."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_CRITERIA = {
    "collision_streak_min": 2,
    "stuck_forward_window": 3,
    "stuck_displacement_m": 0.05,
    "oscillation_turn_count": 6,
    "oscillation_max_step_span": 8,
    "no_progress_window": 12,
    "no_progress_min_forward_actions": 4,
    "no_progress_min_path_m": 0.5,
    "no_progress_distance_delta_m": 0.05,
}


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every random source used by the evaluator and diffusion policy."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_episode_seed(base_seed: int, scene_id: str, episode_id: Any) -> int:
    """Derive a stable per-episode seed without relying on randomized Python hashes."""
    identity = f"{scene_id}:{episode_id}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")
    return (int(base_seed) + offset) % (2**31 - 1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def depth_statistics(depth_m: Any) -> Dict[str, Optional[float]]:
    """Return compact full-frame and center-crop depth statistics in metres."""
    array = np.asarray(depth_m, dtype=np.float32).squeeze()
    if array.ndim != 2:
        return {}

    def summarize(values: np.ndarray, prefix: str) -> Dict[str, Optional[float]]:
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            return {
                f"{prefix}_min_m": None,
                f"{prefix}_p10_m": None,
                f"{prefix}_median_m": None,
                f"{prefix}_near_0_5_ratio": None,
                f"{prefix}_near_1_0_ratio": None,
            }
        return {
            f"{prefix}_min_m": float(np.min(values)),
            f"{prefix}_p10_m": float(np.percentile(values, 10)),
            f"{prefix}_median_m": float(np.median(values)),
            f"{prefix}_near_0_5_ratio": float(np.mean(values < 0.5)),
            f"{prefix}_near_1_0_ratio": float(np.mean(values < 1.0)),
        }

    height, width = array.shape
    y0, y1 = int(height * 0.35), max(int(height * 0.65), int(height * 0.35) + 1)
    x0, x1 = int(width * 0.35), max(int(width * 0.65), int(width * 0.35) + 1)
    stats = summarize(array.reshape(-1), "depth")
    stats.update(summarize(array[y0:y1, x0:x1].reshape(-1), "depth_center"))
    return stats


def project_pixel_point(
    point: Iterable[float],
    source_size: Iterable[int],
    target_size: Iterable[int],
) -> tuple[int, int]:
    """Scale a pixel point between image spaces without changing its meaning."""
    x, y = (float(value) for value in point)
    source_width, source_height = (max(int(value), 1) for value in source_size)
    target_width, target_height = (max(int(value), 1) for value in target_size)
    projected_x = x * max(target_width - 1, 0) / max(source_width - 1, 1)
    projected_y = y * max(target_height - 1, 0) / max(source_height - 1, 1)
    return int(round(projected_x)), int(round(projected_y))


def depth_at_pixel(
    depth_m: Any,
    point: Iterable[float],
    image_size: Iterable[int],
    radius: int = 2,
) -> Optional[float]:
    """Return robust local depth at a point expressed in another image size."""
    depth = np.asarray(depth_m, dtype=np.float32).squeeze()
    if depth.ndim != 2 or depth.size == 0:
        return None
    image_width, image_height = (max(int(value), 1) for value in image_size)
    depth_x, depth_y = project_pixel_point(
        point,
        (image_width, image_height),
        (depth.shape[1], depth.shape[0]),
    )
    if not (0 <= depth_x < depth.shape[1] and 0 <= depth_y < depth.shape[0]):
        return None
    radius = max(int(radius), 0)
    patch = depth[
        max(depth_y - radius, 0) : min(depth_y + radius + 1, depth.shape[0]),
        max(depth_x - radius, 0) : min(depth_x + radius + 1, depth.shape[1]),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(valid)) if valid.size else None


class DiagnosticLogger:
    """Write an episode timeline and derive conservative navigation failure signals."""

    def __init__(
        self,
        root_dir: str,
        run_metadata: Dict[str, Any],
        criteria: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.criteria = dict(DEFAULT_CRITERIA)
        if criteria:
            self.criteria.update(criteria)

        metadata = {
            **run_metadata,
            "criteria": self.criteria,
            "created_unix_s": time.time(),
        }
        self._write_json(self.root_dir / "run.json", metadata)
        self.episode_dir: Optional[Path] = None
        self.timeline_path: Optional[Path] = None
        self.reset_episode_state()

    def reset_episode_state(self) -> None:
        self.sequence_id = 0
        self.collision_steps = 0
        self.collision_streak_events = 0
        self.current_collision_streak = 0
        self.max_collision_streak = 0
        self.stuck_events = 0
        self.oscillation_events = 0
        self.no_progress_events = 0
        self.s2_calls = 0
        self.s1_plans = 0
        self.pixel_goal_calls = 0
        self._recent_actions: deque = deque(maxlen=max(16, int(self.criteria["no_progress_window"])))
        self._recent_turns: deque = deque(maxlen=int(self.criteria["oscillation_turn_count"]))
        self._last_stuck_step = -1
        self._last_oscillation_step = -1
        self._last_no_progress_step = -1

    def start_episode(self, episode_metadata: Dict[str, Any]) -> Path:
        self.reset_episode_state()
        scene = str(episode_metadata["scene_id"])
        episode = str(episode_metadata["episode_id"])
        safe_scene = scene.replace("/", "_").replace("\\", "_")
        safe_episode = episode.replace("/", "_").replace("\\", "_")
        self.episode_dir = self.root_dir / "episodes" / f"{safe_scene}_{safe_episode}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "depth").mkdir(exist_ok=True)
        (self.episode_dir / "plans").mkdir(exist_ok=True)
        (self.episode_dir / "s2_decisions").mkdir(exist_ok=True)
        self.timeline_path = self.episode_dir / "timeline.jsonl"
        self.timeline_path.write_text("", encoding="utf-8")
        self._write_json(self.episode_dir / "episode.json", episode_metadata)
        self.log("episode_start", **episode_metadata)
        return self.episode_dir

    def save_s2_decision(
        self,
        s2_call_id: int,
        image: Any,
        *,
        decision_step: int,
        output_type: str,
        raw_output: str,
        pixel_goal: Optional[Iterable[int]] = None,
        action_names: Optional[Iterable[str]] = None,
        current_subtask: Optional[str] = None,
        depth_m: Any = None,
        model_image_size: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        """Save the exact current S2 image with a diagnostic-only decision overlay."""
        if self.episode_dir is None:
            raise RuntimeError("start_episode must be called before saving an S2 decision")
        if isinstance(image, Image.Image):
            decision_image = image.convert("RGB")
        else:
            decision_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        if model_image_size is not None:
            model_size = tuple(int(value) for value in model_image_size)
            if decision_image.size != model_size:
                decision_image = decision_image.resize(model_size)

        point = [int(value) for value in pixel_goal] if pixel_goal is not None else None
        point_depth_m = (
            depth_at_pixel(depth_m, point, decision_image.size)
            if point is not None and depth_m is not None
            else None
        )
        action_names = [str(value) for value in (action_names or [])]
        safe_output = str(raw_output).encode("ascii", errors="replace").decode("ascii")
        safe_subtask = str(current_subtask or "unknown").encode("ascii", errors="replace").decode("ascii")

        draw = ImageDraw.Draw(decision_image, "RGBA")
        draw.rectangle((0, 0, decision_image.width, 58), fill=(0, 0, 0, 190))
        draw.text(
            (8, 7),
            f"S2 #{int(s2_call_id)}  step {int(decision_step)}  {str(output_type).upper()}",
            fill=(255, 255, 255, 255),
        )
        draw.text((8, 24), f"output: {safe_output[:62]}", fill=(255, 255, 0, 255))
        draw.text((8, 41), f"subtask: {safe_subtask[:60]}", fill=(210, 220, 230, 255))

        if point is not None:
            x, y = point
            colour = (0, 255, 255, 255)
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=(0, 0, 0, 255), width=5)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=colour, width=3)
            draw.line((x - 14, y, x + 14, y), fill=colour, width=2)
            draw.line((x, y - 14, x, y + 14), fill=colour, width=2)
            depth_label = "invalid" if point_depth_m is None else f"{point_depth_m:.2f} m"
            draw.rectangle((x + 12, y - 12, x + 116, y + 10), fill=(0, 0, 0, 190))
            draw.text((x + 16, y - 9), f"GOAL {x},{y}  {depth_label}", fill=colour)
        elif "STOP" in action_names or str(output_type).startswith("stop"):
            draw.rectangle(
                (
                    decision_image.width // 2 - 72,
                    decision_image.height // 2 - 27,
                    decision_image.width // 2 + 72,
                    decision_image.height // 2 + 27,
                ),
                fill=(160, 0, 0, 210),
                outline=(255, 80, 80, 255),
                width=3,
            )
            draw.text(
                (decision_image.width // 2 - 22, decision_image.height // 2 - 6),
                "STOP",
                fill=(255, 255, 255, 255),
            )
        elif action_names:
            label = " | ".join(action_names)
            draw.rectangle(
                (24, decision_image.height - 43, decision_image.width - 24, decision_image.height - 12),
                fill=(0, 0, 0, 190),
                outline=(255, 210, 0, 255),
                width=2,
            )
            draw.text((34, decision_image.height - 33), label[:52], fill=(255, 220, 0, 255))

        file_path = self.episode_dir / "s2_decisions" / f"decision_{int(s2_call_id):04d}.jpg"
        decision_image.save(file_path, format="JPEG", quality=90, optimize=True)
        return {
            "decision_image": str(file_path.relative_to(self.root_dir)),
            "decision_image_size": list(decision_image.size),
            "decision_point": point,
            "decision_point_depth_m": point_depth_m,
            "decision_action_names": action_names,
            "decision_current_subtask": current_subtask,
        }

    def log(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        if self.timeline_path is None:
            raise RuntimeError("start_episode must be called before logging")
        event = {
            "sequence_id": self.sequence_id,
            "event_type": event_type,
            "time_unix_s": time.time(),
            **fields,
        }
        self.sequence_id += 1
        event = _jsonable(event)
        with self.timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
        if event_type == "s2_inference":
            self.s2_calls += 1
            if fields.get("output_type") == "pixel_goal":
                self.pixel_goal_calls += 1
        elif event_type == "s1_plan":
            self.s1_plans += 1
        return event

    def save_depth(self, label: str, depth_m: Any, **metadata: Any) -> Dict[str, Any]:
        if self.episode_dir is None:
            raise RuntimeError("start_episode must be called before saving depth")
        array = np.asarray(depth_m, dtype=np.float32).squeeze()
        file_path = self.episode_dir / "depth" / f"{label}.npz"
        np.savez_compressed(file_path, depth_m=array)
        result = {
            "depth_file": str(file_path.relative_to(self.root_dir)),
            **depth_statistics(array),
            **metadata,
        }
        self.log("depth_snapshot", label=label, **result)
        return result

    def save_s1_plan(self, plan_id: int, dp_actions: Any, **metadata: Any) -> Dict[str, Any]:
        if self.episode_dir is None:
            raise RuntimeError("start_episode must be called before saving an S1 plan")
        if torch.is_tensor(dp_actions):
            raw = dp_actions.detach().float().cpu().numpy().copy()
        else:
            raw = np.asarray(dp_actions, dtype=np.float32).copy()
        if raw.ndim != 3 or raw.shape[-1] < 2:
            raise ValueError(f"Expected S1 trajectories shaped [N, T, >=2], got {raw.shape}")

        scaled = raw.copy()
        scaled[:, :, :2] /= 4.0
        cumulative_xy = np.concatenate(
            [np.zeros((scaled.shape[0], 1, 2), dtype=np.float32), np.cumsum(scaled[:, :, :2], axis=1)],
            axis=1,
        )
        mean_xy = np.mean(cumulative_xy, axis=0)
        endpoint_spread_m = float(np.mean(np.linalg.norm(cumulative_xy[:, -1] - mean_xy[-1], axis=1)))
        file_path = self.episode_dir / "plans" / f"plan_{int(plan_id):04d}.npz"
        np.savez_compressed(
            file_path,
            raw_delta_xyt=raw,
            scaled_delta_xyt=scaled,
            candidate_xy=cumulative_xy,
            mean_xy=mean_xy,
        )
        summary = {
            "plan_id": int(plan_id),
            "plan_file": str(file_path.relative_to(self.root_dir)),
            "candidate_count": int(raw.shape[0]),
            "trajectory_steps": int(raw.shape[1]),
            "mean_endpoint_xy_m": mean_xy[-1].tolist(),
            "endpoint_spread_m": endpoint_spread_m,
            **metadata,
        }
        self.log("s1_plan", **summary)
        return summary

    def record_env_step(
        self,
        *,
        env_step: int,
        decision_step: int,
        action: int,
        action_name: str,
        gps_before: Iterable[float],
        gps_after: Iterable[float],
        distance_before: Optional[float],
        distance_after: Optional[float],
        collision: bool,
        collision_count: Optional[int],
        **fields: Any,
    ) -> Dict[str, Any]:
        gps_before_array = np.asarray(gps_before, dtype=np.float32)
        gps_after_array = np.asarray(gps_after, dtype=np.float32)
        displacement = float(np.linalg.norm(gps_after_array - gps_before_array))

        record = {
            "env_step": int(env_step),
            "decision_step": int(decision_step),
            "action": int(action),
            "action_name": action_name,
            "gps_before": gps_before_array.tolist(),
            "gps_after": gps_after_array.tolist(),
            "displacement_m": displacement,
            "distance_before_m": distance_before,
            "distance_after_m": distance_after,
            "collision": bool(collision),
            "collision_count": collision_count,
        }

        flags = []
        is_navigation_action = action_name in {"FORWARD", "LEFT", "RIGHT"}
        if is_navigation_action:
            if collision:
                self.collision_steps += 1
                self.current_collision_streak += 1
                self.max_collision_streak = max(self.max_collision_streak, self.current_collision_streak)
                if self.current_collision_streak == int(self.criteria["collision_streak_min"]):
                    self.collision_streak_events += 1
            else:
                self.current_collision_streak = 0
            self._recent_actions.append(record)

        forward_window = int(self.criteria["stuck_forward_window"])
        recent_forward = list(self._recent_actions)[-forward_window:]
        if (
            len(recent_forward) == forward_window
            and all(item["action_name"] == "FORWARD" for item in recent_forward)
            and decision_step - self._last_stuck_step >= forward_window
        ):
            total_displacement = float(
                np.linalg.norm(
                    np.asarray(recent_forward[-1]["gps_after"], dtype=np.float32)
                    - np.asarray(recent_forward[0]["gps_before"], dtype=np.float32)
                )
            )
            if total_displacement < float(self.criteria["stuck_displacement_m"]):
                self.stuck_events += 1
                self._last_stuck_step = decision_step
                flags.append("stuck")

        if action_name in {"LEFT", "RIGHT"}:
            self._recent_turns.append((decision_step, action_name))
            required_turns = int(self.criteria["oscillation_turn_count"])
            turns = list(self._recent_turns)
            alternating = len(turns) == required_turns and all(
                turns[index][1] != turns[index - 1][1] for index in range(1, len(turns))
            )
            if (
                alternating
                and turns[-1][0] - turns[0][0] <= int(self.criteria["oscillation_max_step_span"])
                and decision_step - self._last_oscillation_step >= required_turns
            ):
                self.oscillation_events += 1
                self._last_oscillation_step = decision_step
                flags.append("oscillation")

        no_progress_window = int(self.criteria["no_progress_window"])
        recent_progress = list(self._recent_actions)[-no_progress_window:]
        if (
            len(recent_progress) == no_progress_window
            and recent_progress[0]["distance_before_m"] is not None
            and recent_progress[-1]["distance_after_m"] is not None
            and decision_step - self._last_no_progress_step >= no_progress_window
        ):
            forward_actions = sum(item["action_name"] == "FORWARD" for item in recent_progress)
            path_length = float(sum(item["displacement_m"] for item in recent_progress))
            improvement = float(
                recent_progress[0]["distance_before_m"] - recent_progress[-1]["distance_after_m"]
            )
            if (
                forward_actions >= int(self.criteria["no_progress_min_forward_actions"])
                and path_length >= float(self.criteria["no_progress_min_path_m"])
                and improvement < float(self.criteria["no_progress_distance_delta_m"])
            ):
                self.no_progress_events += 1
                self._last_no_progress_step = decision_step
                flags.append("no_progress")
                record["no_progress_evidence"] = {
                    "window": no_progress_window,
                    "forward_actions": forward_actions,
                    "path_length_m": path_length,
                    "ne_improvement_m": improvement,
                }

        return self.log("env_step", **record, flags=flags, **fields)

    def finish_episode(self, metrics: Dict[str, Any], stop_reason: str, **fields: Any) -> Dict[str, Any]:
        success = float(metrics.get("success", 0.0))
        low_level_failure_signal = any(
            [
                self.collision_streak_events > 0,
                self.stuck_events > 0,
                self.oscillation_events > 0,
                self.no_progress_events > 0,
            ]
        )
        automatic_candidate = success == 0.0 and self.pixel_goal_calls > 0 and low_level_failure_signal
        summary = {
            "metrics": metrics,
            "stop_reason": stop_reason,
            "s2_calls": self.s2_calls,
            "s1_plans": self.s1_plans,
            "pixel_goal_calls": self.pixel_goal_calls,
            "collision_steps": self.collision_steps,
            "collision_streak_events": self.collision_streak_events,
            "max_collision_streak": self.max_collision_streak,
            "stuck_events": self.stuck_events,
            "oscillation_events": self.oscillation_events,
            "no_progress_events": self.no_progress_events,
            "automatic_s1_failure_candidate": automatic_candidate,
            "pure_s1_failure": None,
            "manual_review_required": automatic_candidate,
            **fields,
        }
        self.log("episode_end", **summary)
        if self.episode_dir is None:
            raise RuntimeError("start_episode must be called before finishing an episode")
        self._write_json(self.episode_dir / "summary.json", summary)
        return _jsonable(summary)

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_path, path)
