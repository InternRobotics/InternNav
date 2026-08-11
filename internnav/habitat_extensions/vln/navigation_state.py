"""Lightweight structured context for S2 navigation decisions.

The helpers in this module deliberately avoid learned components.  They turn
signals already produced by Habitat (instruction text, depth, pose, and recent
goals) into compact state that can be logged, unit-tested, and optionally added
to the S2 prompt.
"""

from collections import deque
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


_ACTION_START = (
    "turn|walk|go|head|move|exit|enter|stop|wait|step|continue|take|follow|"
    "face|look|proceed|pass|cross|climb|descend"
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "tenth": 10,
}


@dataclass
class CameraPitchState:
    """Track temporary low-head camera actions in 15-degree Habitat steps."""

    down_steps: int = 0

    def record_look_down(self, count=1):
        self.down_steps += max(0, int(count))

    def record_look_up(self, count=1):
        self.down_steps = max(0, self.down_steps - max(0, int(count)))

    @property
    def restore_steps(self):
        return self.down_steps

    def mark_horizontal(self):
        self.down_steps = 0


def bound_actions_at_lookdown(actions, lookdown_action=5):
    """Make LOOKDOWN an observation boundary, not a queued persistent tilt."""
    sequence = [int(action) for action in actions]
    try:
        index = sequence.index(int(lookdown_action))
    except ValueError:
        return sequence
    return sequence[: index + 1]


def split_instruction(instruction: str) -> tuple[str, ...]:
    """Split an R2R instruction into conservative action-sized clauses."""
    text = re.sub(r"\s+", " ", str(instruction).strip()).strip(" .")
    if not text:
        return ()
    text = re.sub(r"\b(?:and then|then|after that|afterwards|next)\b", "|", text, flags=re.I)
    text = re.sub(r"\b(?:once you|when you)\b", "|", text, flags=re.I)
    text = re.sub(r"[.;]+", "|", text)
    text = re.sub(r",\s*(?:and\s+)?(?=(?:" + _ACTION_START + r")\b)", "|", text, flags=re.I)
    text = re.sub(r"\s+and\s+(?=(?:" + _ACTION_START + r")\b)", "|", text, flags=re.I)
    parts = [part.strip(" ,.") for part in text.split("|") if part.strip(" ,.")]
    merged = []
    for part in parts:
        if (
            merged
            and re.match(
                r"^stop\s+(?:at|on)\s+the\s+(?:very\s+)?(?:top|bottom|landing)\b",
                part,
                re.I,
            )
            and re.search(
                r"\b(stair|stairs|stairway|step|steps|upstairs|downstairs|climb|descend)\b",
                merged[-1],
                re.I,
            )
        ):
            merged[-1] = f"{merged[-1]} and {part}"
        else:
            merged.append(part)
    return tuple(merged)


def select_history_indices(frame_count: int, num_history: int, prioritize_recent: bool = False) -> list[int]:
    """Select prior RGB frames while excluding the current final frame."""
    previous_count = max(0, int(frame_count) - 1)
    count = min(max(0, int(num_history)), previous_count)
    if count == 0:
        return []
    if count == previous_count:
        return list(range(previous_count))

    recent_fraction = 0.75 if prioritize_recent else 0.5
    recent_count = min(count, max(1, int(np.ceil(count * recent_fraction))))
    recent = list(range(previous_count - recent_count, previous_count))
    older_count = count - len(recent)
    older_stop = previous_count - recent_count
    older = []
    if older_count > 0 and older_stop > 0:
        older = np.unique(np.linspace(0, older_stop - 1, older_count, dtype=np.int32)).tolist()
    selected = sorted(set(int(index) for index in older + recent))
    if len(selected) < count:
        for index in range(previous_count - 1, -1, -1):
            if index not in selected:
                selected.append(index)
            if len(selected) == count:
                break
    return sorted(selected[-count:] if len(selected) > count else selected)


def select_uniform_history_indices(frame_count: int, num_history: int) -> list[int]:
    """Restore the original evenly spaced history without semantic recency bias."""
    previous_count = max(0, int(frame_count) - 1)
    count = min(max(0, int(num_history)), previous_count)
    if count == 0:
        return []
    if count == previous_count:
        return list(range(previous_count))
    return np.unique(
        np.linspace(0, previous_count - 1, count, dtype=np.int32)
    ).tolist()


@dataclass(frozen=True)
class DepthSummary:
    text: str
    left_clearance: Optional[float]
    centre_clearance: Optional[float]
    right_clearance: Optional[float]
    invalid_fraction: float
    centre_open: bool
    centre_blocked: bool
    step_edge_count: int


class DepthObservationSummarizer:
    """Summarize a low-head depth frame without projecting or building a map."""

    def __init__(self, near_distance=0.55, open_margin=0.25):
        self.near_distance = float(near_distance)
        self.open_margin = float(open_margin)

    @staticmethod
    def _clearance(region):
        values = region[np.isfinite(region) & (region > 0.05)]
        if values.size == 0:
            return None
        return float(np.quantile(values, 0.15))

    def summarize(self, depth, stair_mode=False) -> DepthSummary:
        array = np.asarray(depth, dtype=np.float32).squeeze()
        if array.ndim != 2:
            raise ValueError("depth must be a 2-D image in metres")
        height, width = array.shape
        crop = array[int(height * 0.18) : max(int(height * 0.95), 1), int(width * 0.05) : int(width * 0.95)]
        valid = np.isfinite(crop) & (crop > 0.05)
        invalid_fraction = float(1.0 - np.mean(valid)) if crop.size else 1.0
        thirds = np.array_split(crop, 3, axis=1)
        clearances = [self._clearance(region) for region in thirds]
        left, centre, right = clearances

        centre_open = bool(
            centre is not None
            and left is not None
            and right is not None
            and centre >= max(left, right) + self.open_margin
        )
        centre_values = thirds[1][np.isfinite(thirds[1]) & (thirds[1] > 0.05)]
        centre_near_fraction = (
            float(np.mean(centre_values < self.near_distance)) if centre_values.size else 1.0
        )
        centre_blocked = bool(centre is not None and centre < self.near_distance and centre_near_fraction >= 0.15)

        centre_strip = crop[:, crop.shape[1] // 3 : (2 * crop.shape[1]) // 3]
        row_medians = []
        for band in np.array_split(centre_strip, 8, axis=0):
            values = band[np.isfinite(band) & (band > 0.05)]
            row_medians.append(float(np.median(values)) if values.size else np.nan)
        row_medians = np.asarray(row_medians, dtype=np.float32)
        finite_pairs = np.isfinite(row_medians[:-1]) & np.isfinite(row_medians[1:])
        changes = np.abs(np.diff(row_medians))[finite_pairs]
        step_edge_count = int(np.sum((changes >= 0.12) & (changes <= 1.25)))

        def render(value):
            return "unknown" if value is None else f"{value:.2f} m"

        statements = [
            "Low-head geometric depth summary",
            f"left/centre/right near clearance is {render(left)} / {render(centre)} / {render(right)}",
        ]
        if invalid_fraction >= 0.35:
            statements.append(
                f"depth is unreliable in {invalid_fraction:.0%} of the inspected area, so verify with RGB"
            )
        if centre_open:
            statements.append("the centre is more open than both sides")
            if stair_mode:
                statements.append(
                    "side obstacles may be railings; prefer the open centre corridor and do not aim at the sides"
                )
        elif centre_blocked:
            statements.append("the centre has a near obstacle; do not move forward without visual confirmation")
        if stair_mode and step_edge_count >= 2:
            statements.append(
                "the centre depth has repeated discontinuities consistent with steps, but this is not semantic proof"
            )
        statements.append("depth describes geometry only and cannot by itself identify stairs or glass")
        return DepthSummary(
            text="; ".join(statements) + ".",
            left_clearance=left,
            centre_clearance=centre,
            right_clearance=right,
            invalid_fraction=invalid_fraction,
            centre_open=centre_open,
            centre_blocked=centre_blocked,
            step_edge_count=step_edge_count,
        )


@dataclass(frozen=True)
class PixelGoalDecision:
    duplicate: bool
    failed_duplicate_count: int
    reject: bool
    pixel_distance: Optional[float]
    pose_distance: Optional[float]
    heading_difference_deg: Optional[float]


class PixelGoalMemory:
    """Detect failed near-duplicate S2 pixel goals from nearly the same pose."""

    def __init__(self, pixel_tolerance=32.0, pose_tolerance=0.30, heading_tolerance_deg=20.0, retry_limit=2):
        self.pixel_tolerance = float(pixel_tolerance)
        self.pose_tolerance = float(pose_tolerance)
        self.heading_tolerance_rad = np.deg2rad(float(heading_tolerance_deg))
        self.retry_limit = int(retry_limit)
        self.reset()

    def reset(self):
        self.last_goal = None
        self.last_gps = None
        self.last_compass = None
        self.failed_duplicate_count = 0

    @staticmethod
    def _compass_value(compass):
        values = np.asarray(compass, dtype=np.float32).reshape(-1)
        return float(values[0]) if values.size and np.isfinite(values[0]) else None

    def evaluate(self, pixel_goal, gps, compass, previous_goal_failed=False):
        goal = np.asarray(pixel_goal, dtype=np.float32).reshape(-1)[:2]
        position = np.asarray(gps, dtype=np.float32).reshape(-1)[:2]
        heading = self._compass_value(compass)
        pixel_distance = pose_distance = heading_difference = None
        duplicate = False
        if self.last_goal is not None:
            pixel_distance = float(np.linalg.norm(goal - self.last_goal))
            if self.last_gps is not None and position.size == 2 and np.all(np.isfinite(position)):
                pose_distance = float(np.linalg.norm(position - self.last_gps))
            if self.last_compass is not None and heading is not None:
                delta = np.arctan2(np.sin(heading - self.last_compass), np.cos(heading - self.last_compass))
                heading_difference = float(abs(delta))
            duplicate = bool(
                pixel_distance <= self.pixel_tolerance
                and pose_distance is not None
                and pose_distance <= self.pose_tolerance
                and heading_difference is not None
                and heading_difference <= self.heading_tolerance_rad
            )

        if duplicate and previous_goal_failed:
            self.failed_duplicate_count += 1
        elif not duplicate:
            self.failed_duplicate_count = 0
        reject = bool(duplicate and previous_goal_failed and self.failed_duplicate_count >= self.retry_limit)
        if not reject:
            self.last_goal = goal.copy()
            self.last_gps = position.copy() if position.size == 2 and np.all(np.isfinite(position)) else None
            self.last_compass = heading
        return PixelGoalDecision(
            duplicate=duplicate,
            failed_duplicate_count=self.failed_duplicate_count,
            reject=reject,
            pixel_distance=pixel_distance,
            pose_distance=pose_distance,
            heading_difference_deg=(
                float(np.rad2deg(heading_difference)) if heading_difference is not None else None
            ),
        )


class InstructionStateTracker:
    """Conservative instruction state with explicit stair-height evidence."""

    def __init__(self, instruction, initial_height=0.0, initial_compass=0.0, initial_gps=None):
        self.instruction = str(instruction)
        self.clauses = split_instruction(self.instruction) or (self.instruction.strip(),)
        self.current_index = 0
        self.completed = []
        self.completion_reasons = []
        self.current_start_height = float(initial_height)
        self.current_start_compass = float(initial_compass)
        self.current_start_gps = self._gps(initial_gps)
        self.current_height = float(initial_height)
        self.current_compass = float(initial_compass)
        self.current_gps = self.current_start_gps
        self.height_history = deque([float(initial_height)], maxlen=6)
        self.stop_rejections = 0

    @staticmethod
    def _gps(gps):
        values = np.asarray(gps if gps is not None else [np.nan, np.nan], dtype=np.float32).reshape(-1)
        if values.size < 2 or not np.all(np.isfinite(values[:2])):
            return None
        return values[:2].copy()

    @property
    def current_clause(self):
        return self.clauses[self.current_index] if self.current_index < len(self.clauses) else None

    @property
    def remaining(self):
        return self.clauses[self.current_index :]

    @property
    def all_complete(self):
        return self.current_index >= len(self.clauses)

    @staticmethod
    def is_stair_clause(clause):
        return bool(clause and re.search(r"\b(stair|stairs|stairway|step|steps|upstairs|downstairs|climb|descend)\b", clause, re.I))

    @property
    def stair_mode(self):
        return self.is_stair_clause(self.current_clause)

    @staticmethod
    def _stair_direction(clause):
        if not clause:
            return None
        if re.search(r"\b(down|descend|downstairs|bottom)\b", clause, re.I):
            return "down"
        if re.search(r"\b(up|climb|upstairs|top)\b", clause, re.I):
            return "up"
        return None

    @staticmethod
    def _step_count(clause):
        if not clause:
            return None
        match = re.search(r"\b(\d+)\s*(?:stair|stairs|step|steps)\b", clause, re.I)
        if match:
            return int(match.group(1))
        lowered = clause.lower()
        for word, value in _NUMBER_WORDS.items():
            if re.search(rf"\b{word}\s+(?:stair|stairs|step|steps)\b", lowered):
                return value
        return None

    def _stair_required_height(self, clause):
        count = self._step_count(clause)
        if count is not None:
            return max(0.18, min(1.8, 0.14 * count))
        return 0.45

    def _advance(self, reason):
        if self.all_complete:
            return False
        self.completed.append(self.current_clause)
        self.completion_reasons.append(str(reason))
        self.current_index += 1
        self.current_start_height = self.current_height
        self.current_start_compass = self.current_compass
        self.current_start_gps = None if self.current_gps is None else self.current_gps.copy()
        self.height_history.clear()
        self.height_history.append(self.current_height)
        return True

    def observe(self, height, compass, gps=None):
        self.current_height = float(height)
        compass_values = np.asarray(compass, dtype=np.float32).reshape(-1)
        if compass_values.size and np.isfinite(compass_values[0]):
            self.current_compass = float(compass_values[0])
        self.current_gps = self._gps(gps)
        self.height_history.append(self.current_height)
        clause = self.current_clause
        if clause is None:
            return None

        stair_direction = self._stair_direction(clause) if self.is_stair_clause(clause) else None
        if stair_direction:
            signed_delta = self.current_height - self.current_start_height
            progress = signed_delta if stair_direction == "up" else -signed_delta
            required = self._stair_required_height(clause)
            height_met = progress >= required
            endpoint_required = bool(re.search(r"\b(top|bottom|end|landing)\b", clause, re.I))
            plateau = len(self.height_history) >= 4 and np.ptp(np.asarray(self.height_history)[-4:]) <= 0.08
            if height_met and (not endpoint_required or plateau):
                self._advance("vertical_height_and_plateau" if endpoint_required else "vertical_height")
                return "stair_complete"

        lowered = clause.lower()
        pure_turn = bool(re.match(r"^turn\s+(?:left|right|around)\b", lowered)) and not re.search(
            r"\b(walk|go|move|enter|exit|step|stairs?)\b", lowered
        )
        if pure_turn:
            delta = np.arctan2(
                np.sin(self.current_compass - self.current_start_compass),
                np.cos(self.current_compass - self.current_start_compass),
            )
            required = np.deg2rad(140.0 if "around" in lowered else 60.0)
            if abs(delta) >= required:
                self._advance("heading_change")
                return "turn_complete"
        return None

    def mark_model_completion(self, output):
        if re.search(r"\b(?:SUBTASK_DONE|CURRENT_SUBTASK_DONE)\b", str(output), re.I):
            if self.stair_mode:
                return False
            return self._advance("s2_explicit_marker")
        return False

    def unresolved_stair_clauses(self):
        return [clause for clause in self.remaining if self.is_stair_clause(clause)]

    def _current_stair_evidence(self):
        clause = self.current_clause
        direction = self._stair_direction(clause) if self.is_stair_clause(clause) else None
        if direction is None:
            return None
        signed_delta = self.current_height - self.current_start_height
        progress = signed_delta if direction == "up" else -signed_delta
        endpoint_required = bool(re.search(r"\b(top|bottom|end|landing)\b", clause, re.I))
        plateau = len(self.height_history) >= 4 and np.ptp(np.asarray(self.height_history)[-4:]) <= 0.08
        return {
            "progress": float(progress),
            "required": float(self._stair_required_height(clause)),
            "endpoint_required": endpoint_required,
            "plateau": bool(plateau),
        }

    def terminal_stop_evidence(self):
        """Return whether height tracking found a terminal stair endpoint."""
        return bool(
            self.all_complete
            and self.completion_reasons
            and self.completion_reasons[-1] == "vertical_height_and_plateau"
        )

    def should_force_stop(self, stall_confirmed=False):
        """Force STOP only after endpoint evidence and an independent turn stall."""
        return bool(self.terminal_stop_evidence() and stall_confirmed)

    def should_reject_stop(self):
        """Reject STOP while ordered stair completion still lacks evidence."""
        if self.all_complete or not self.unresolved_stair_clauses():
            return False
        self.stop_rejections += 1
        evidence = self._current_stair_evidence()
        if evidence is None:
            # A future stair clause is still unresolved, so accepting STOP would
            # skip at least one ordered subtask.
            return True
        if evidence["progress"] < evidence["required"]:
            return True
        return bool(evidence["endpoint_required"] and not evidence["plateau"])

    def prompt_context(self):
        completed = " | ".join(self.completed) if self.completed else "none confirmed"
        current = self.current_clause or "all parsed subtasks are complete"
        remaining = " | ".join(self.remaining[1:]) if len(self.remaining) > 1 else "none after current"
        pieces = [
            "Controller-maintained instruction state (verify against the images)",
            f"confirmed completed: {completed}",
            f"current subtask: {current}",
            f"later subtasks: {remaining}",
        ]
        if self.stair_mode:
            direction = self._stair_direction(self.current_clause) or "unknown"
            delta = self.current_height - self.current_start_height
            pieces.append(
                f"current stair direction is {direction}; measured vertical change since this subtask began is {delta:+.2f} m"
            )
            pieces.append("do not treat a side railing as the traversable stair corridor")
        pieces.append(
            "If the current subtask has just been completed, include SUBTASK_DONE in the reply before selecting the next waypoint"
        )
        pieces.append("output STOP only when the full instruction is complete")
        return "; ".join(pieces) + "."

    def semantic_status(self, metric_success, manual_label=None):
        if manual_label and manual_label.get("semantic_status"):
            return str(manual_label["semantic_status"])
        if not bool(metric_success):
            return "metric_failure"
        if self.all_complete:
            return "heuristic_complete"
        if self.unresolved_stair_clauses():
            return "metric_success_stair_unverified"
        return "metric_success_semantically_unverified"

    def as_dict(self):
        return {
            "clauses": list(self.clauses),
            "completed": list(self.completed),
            "completion_reasons": list(self.completion_reasons),
            "current_subtask": self.current_clause,
            "remaining": list(self.remaining),
            "all_complete": self.all_complete,
            "stair_mode": self.stair_mode,
            "height_change_m": self.current_height - self.current_start_height,
            "stop_rejections": self.stop_rejections,
            "terminal_stop_evidence": self.terminal_stop_evidence(),
            "force_stop_ready": False,
        }


def load_semantic_labels(path):
    if not path:
        return {}
    label_path = Path(path)
    if not label_path.exists():
        return {}
    with label_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def semantic_label_for_episode(labels, scene_id, episode_id):
    keys = (f"{scene_id}_{episode_id}", f"{scene_id}/{episode_id}", str(episode_id))
    for key in keys:
        value = labels.get(key)
        if isinstance(value, dict):
            return value
    return {}
