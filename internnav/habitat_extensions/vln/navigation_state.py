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


@dataclass(frozen=True)
class ConfidenceGateDecision:
    reject: bool
    reason: Optional[str] = None
    confirmed: bool = False


class S2ConfidenceGate:
    """Conservative confidence gates for pixel goals and model STOP outputs.

    Confidence here is generation confidence, not calibrated semantic
    correctness.  It can reject unusually uncertain outputs, but cannot prove
    that a referenced landmark was identified correctly.
    """

    def __init__(self, pixel_goal_threshold=0.35, stop_threshold=0.75):
        self.pixel_goal_threshold = max(0.0, float(pixel_goal_threshold))
        self.stop_threshold = max(0.0, float(stop_threshold))
        self.pending_low_confidence_stop = False

    def reset(self):
        self.pending_low_confidence_stop = False

    @staticmethod
    def _below(confidence, threshold):
        return confidence is not None and threshold > 0.0 and float(confidence) < threshold

    def evaluate_pixel_goal(self, confidence):
        self.pending_low_confidence_stop = False
        if self._below(confidence, self.pixel_goal_threshold):
            return ConfidenceGateDecision(True, "low_pixel_goal_generation_confidence")
        return ConfidenceGateDecision(False)

    def evaluate_stop(self, confidence):
        if not self._below(confidence, self.stop_threshold):
            self.pending_low_confidence_stop = False
            return ConfidenceGateDecision(False)
        if self.pending_low_confidence_stop:
            self.pending_low_confidence_stop = False
            return ConfidenceGateDecision(False, "repeated_low_confidence_stop", confirmed=True)
        self.pending_low_confidence_stop = True
        return ConfidenceGateDecision(True, "low_stop_generation_confidence")

    def observe_non_stop(self):
        self.pending_low_confidence_stop = False


_RELATIONAL_LANGUAGE = re.compile(
    r"\b(first|second|left of|right of|opposite|past|after|before|between|"
    r"leading into|through|towards?|facing|away from|top of|bottom of)\b",
    re.I,
)


@dataclass(frozen=True)
class SemanticShadowDecision:
    """Diagnostic-only probe decision; it must never change navigation actions."""

    query: bool
    uncertainty_signal: bool
    reasons: tuple[str, ...]
    semantic_risk: bool


class SemanticShadowMonitor:
    """Rate-limit semantic probes and expose uncertainty without action gating."""

    def __init__(
        self,
        generation_threshold=0.55,
        minimum_token_threshold=0.20,
        query_interval=3,
        max_queries=10,
    ):
        self.generation_threshold = float(generation_threshold)
        self.minimum_token_threshold = float(minimum_token_threshold)
        self.query_interval = max(1, int(query_interval))
        self.max_queries = max(0, int(max_queries))
        self.reset()

    def reset(self):
        self.query_count = 0
        self.last_query_call = -10**9
        self.last_clause = None

    @staticmethod
    def _is_stop(output):
        return bool(re.search(r"\bSTOP\b", str(output), re.I))

    def assess(self, call_id, clause, output, generation_confidence, minimum_token_confidence):
        reasons = []
        if (
            generation_confidence is not None
            and float(generation_confidence) < self.generation_threshold
        ):
            reasons.append("low_generation_confidence")
        if (
            minimum_token_confidence is not None
            and float(minimum_token_confidence) < self.minimum_token_threshold
        ):
            reasons.append("low_minimum_token_confidence")

        clause_text = str(clause or "")
        semantic_risk = bool(_RELATIONAL_LANGUAGE.search(clause_text))
        clause_changed = clause_text != self.last_clause
        stop_output = self._is_stop(output)
        uncertainty_signal = bool(reasons)
        due = int(call_id) - self.last_query_call >= self.query_interval
        query = bool(
            self.query_count < self.max_queries
            and due
            and (
                self.query_count == 0
                or clause_changed
                or uncertainty_signal
                or semantic_risk
                or stop_output
            )
        )
        if query:
            self.query_count += 1
            self.last_query_call = int(call_id)
        self.last_clause = clause_text
        return SemanticShadowDecision(
            query=query,
            uncertainty_signal=uncertainty_signal,
            reasons=tuple(reasons),
            semantic_risk=semantic_risk,
        )


@dataclass(frozen=True)
class StageProgressEstimate:
    """Diagnostic estimate of how far S2 has progressed through ordered clauses."""

    controller_index: int
    inferred_index: int
    status: str
    clause_results: tuple[str, ...]


def parse_stage_completion_output(output):
    """Map S2's native navigation vocabulary to one progress judgement.

    STOP means the queried subtask is definitely complete. RIGHT means it is
    definitely not complete, and LEFT is reserved for uncertainty.  Keeping
    the output vocabulary native avoids assuming that the base S2 can reliably
    generate a new JSON schema before any fine-tuning.
    """
    text = re.sub(r"\s+", "", str(output or "")).upper()
    if re.search(r"\bSTOP\b", text):
        return "COMPLETE"
    if "→" in text or text in {"RIGHT", "TURNRIGHT"}:
        return "INCOMPLETE"
    if "←" in text or text in {"LEFT", "TURNLEFT"}:
        return "UNCERTAIN"
    return "UNCERTAIN"


def estimate_stage_progress(controller_index, clause_outputs, clause_count):
    """Advance only across a consecutive prefix of definitely complete clauses."""
    start = max(0, min(int(controller_index), int(clause_count)))
    inferred = start
    parsed = []
    terminal_status = "TASK_COMPLETE" if inferred >= int(clause_count) else "UNCERTAIN"
    for output in clause_outputs:
        result = parse_stage_completion_output(output)
        parsed.append(result)
        if result == "COMPLETE" and inferred < int(clause_count):
            inferred += 1
            terminal_status = "TASK_COMPLETE" if inferred >= int(clause_count) else "COMPLETE_PREFIX"
            continue
        terminal_status = result
        break
    return StageProgressEstimate(
        controller_index=start,
        inferred_index=inferred,
        status=terminal_status,
        clause_results=tuple(parsed),
    )


def build_stage_probe_prompt(instruction, clauses, clause_index, image_count):
    """Build a narrow chronological-image prompt for one subtask completion check."""
    numbered = " ".join(f"[{index + 1}] {clause}" for index, clause in enumerate(clauses))
    target = clauses[int(clause_index)]
    return (
        "You are only verifying navigation progress, not choosing the next action. "
        f"The full task is: '{instruction}'. Its ordered subtasks are: {numbered}. "
        f"Decide whether subtask [{int(clause_index) + 1}] '{target}' was definitely "
        f"completed at or before the last of these {int(image_count)} chronological images. "
        "Keep every modifier such as color, ordinal, direction, and nearby landmark. "
        "A later subtask may be underway even when an external controller reports an older one. "
        "Output exactly STOP if definitely completed, one RIGHT arrow if definitely not "
        "completed, or one LEFT arrow if the images are insufficient or ambiguous."
    )


def _shadow_point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = int(round(float(value[0]))), int(round(float(value[1])))
    except (TypeError, ValueError):
        return None
    return [x, y] if 0 <= x < 640 and 0 <= y < 480 else None


def parse_semantic_shadow_output(output):
    """Parse best-effort JSON from a diagnostic S2 semantic probe."""
    text = str(output).strip()
    payload = {}
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            decoded = json.loads(match.group(0))
            if isinstance(decoded, dict):
                payload = decoded
        except json.JSONDecodeError:
            payload = {}

    def boolean(name):
        value = payload.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"yes", "true"}:
                return True
            if lowered in {"no", "false"}:
                return False
        fallback = re.search(rf"\b{name}\s*:\s*(yes|no|true|false)\b", text, re.I)
        return fallback.group(1).lower() in {"yes", "true"} if fallback else None

    def point(name):
        parsed = _shadow_point(payload.get(name))
        if parsed is not None:
            return parsed
        fallback = re.search(
            rf"\b{name}\s*:\s*[\[(]?\s*(\d{{1,3}})\s*[, ]\s*(\d{{1,3}})",
            text,
            re.I,
        )
        return _shadow_point(fallback.groups()) if fallback else None

    progress = payload.get("progress_step")
    try:
        progress = int(progress) if progress is not None else None
    except (TypeError, ValueError):
        progress = None
    return {
        "uncertain": boolean("uncertain"),
        "anchor": str(payload.get("anchor") or "unknown")[:120],
        "anchor_point": point("anchor_point"),
        "goal_point": point("goal_point"),
        "progress_step": progress,
        "stop_complete": boolean("stop_complete"),
        "reason": str(payload.get("reason") or "")[:500],
        "raw_output": text,
    }


def extract_reference_landmark(clause):
    """Extract a compact landmark phrase for a native-vocabulary shadow probe."""
    text = re.sub(r"\s+", " ", str(clause or "").strip(" .,"))
    if not text:
        return "visible destination"
    patterns = (
        r"\b(?:left|right)\s+of\s+(?:the\s+)?(.+?)(?=\s+(?:and|then|before|after)\b|[,.;]|$)",
        r"\bopposite\s+(?:of\s+)?(?:the\s+)?(.+?)(?=\s+(?:and|then|before|after)\b|[,.;]|$)",
        r"\baway\s+from\s+(?:the\s+)?(.+?)(?=\s+(?:and|then|before|after)\b|[,.;]|$)",
        r"\b(?:past|towards?|facing)\s+(?:the\s+)?(.+?)(?=\s+(?:and|then|before|after)\b|[,.;]|$)",
        r"\b(?:through|enter|into|approach)\s+(?:the\s+)?(.+?)(?=\s+(?:that|which|to|and|then|on)\b|[,.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            landmark = match.group(1).strip(" ,.")
            if landmark:
                return landmark[:120]
    return text[:120]


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
            "current_index": int(self.current_index),
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
