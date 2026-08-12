"""Minimal stateful recovery policy for queued S1 actions."""

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RecoveryDecision:
    cancel_remaining: bool
    replan_system2: bool
    reason: Optional[str]
    gps_displacement: Optional[float]
    retry_count: int
    goal_retry_count: int
    turn_count: int


class RecoveryController:
    def __init__(
        self,
        min_forward_displacement=0.03,
        system2_retry_limit=3,
        max_consecutive_turns=24,
        direct_turn_escape_clearance=0.55,
    ):
        self.min_forward_displacement = float(min_forward_displacement)
        self.system2_retry_limit = int(system2_retry_limit)
        self.max_consecutive_turns = int(max_consecutive_turns)
        self.direct_turn_escape_clearance = float(direct_turn_escape_clearance)
        self.failed_directions = Counter()
        self.failed_route_directions = Counter()
        self.retry_count = 0
        self.goal_retry_count = 0
        self.consecutive_failures = 0
        self.current_route_direction = None
        self.turn_direction = None
        self.consecutive_turns = 0
        self.pending_turn_stall = False

    def reset(self):
        self.failed_directions.clear()
        self.failed_route_directions.clear()
        self.retry_count = 0
        self.current_route_direction = None
        self.turn_direction = None
        self.consecutive_turns = 0
        self.pending_turn_stall = False
        self.start_new_goal()

    def start_new_goal(self, preserve_failures=False):
        if not preserve_failures:
            self.goal_retry_count = 0
            self.consecutive_failures = 0
        self.current_route_direction = None
        self.turn_direction = None
        self.consecutive_turns = 0

    def set_route_direction(self, direction):
        direction = str(direction).lower() if direction is not None else None
        self.current_route_direction = direction if direction in ("left", "straight", "right") else None

    def failed_route_context(self):
        if not self.failed_route_directions:
            return None
        ordered = sorted(self.failed_route_directions.items(), key=lambda item: (-item[1], item[0]))
        attempts = ", ".join(f"{direction} x{count}" for direction, count in ordered)
        return (
            f"Failed route sectors from this episode: {attempts}. Prefer a less-failed visible route, "
            "but returning or reversing remains allowed when the instruction requires correction."
        )

    def exploratory_turn(self):
        """Return LEFT/RIGHT action code using the less-failed route sector."""
        left = self.failed_route_directions.get("left", 0)
        right = self.failed_route_directions.get("right", 0)
        if self.current_route_direction == "left":
            return 3
        if self.current_route_direction == "right":
            return 2
        return 2 if left <= right else 3

    def consume_turn_stall(self):
        """Return a recent full-circle stall once, then clear the signal."""
        pending = self.pending_turn_stall
        self.pending_turn_stall = False
        return pending

    def filter_direct_actions(self, actions, centre_clearance=None, centre_blocked=False):
        """Replace a repeated full-circle S2 turn with one bounded escape action."""
        sequence = [int(action) for action in actions]
        if not sequence or sequence[0] not in (2, 3) or any(
            action != sequence[0] for action in sequence
        ):
            return sequence, None
        failed_turn = sequence[0]
        if self.failed_directions.get(failed_turn, 0) <= 0:
            return sequence, None

        clearance = None
        if centre_clearance is not None:
            try:
                value = float(centre_clearance)
                clearance = value if np.isfinite(value) else None
            except (TypeError, ValueError):
                clearance = None
        if (
            not bool(centre_blocked)
            and clearance is not None
            and clearance >= self.direct_turn_escape_clearance
        ):
            replacement = [1]
            reason = "depth_gated_forward_after_repeated_direct_turn"
        else:
            replacement = [3 if failed_turn == 2 else 2]
            reason = "opposite_probe_after_repeated_direct_turn"

        # Consume one failure marker. After the bounded escape the changed pose
        # receives a fresh normal-turn budget instead of being permanently banned.
        self.failed_directions[failed_turn] -= 1
        if self.failed_directions[failed_turn] <= 0:
            del self.failed_directions[failed_turn]
        return replacement, reason

    def update(self, action, gps_before, gps_after, collision=False, is_s1=True):
        action = int(action)
        before = np.asarray(gps_before, dtype=np.float32).reshape(-1)
        after = np.asarray(gps_after, dtype=np.float32).reshape(-1)
        displacement = None
        if before.size >= 2 and after.size >= 2 and np.all(np.isfinite(before[:2])) and np.all(np.isfinite(after[:2])):
            displacement = float(np.linalg.norm(after[:2] - before[:2]))

        if action in (2, 3) and self.max_consecutive_turns > 0:
            if action == self.turn_direction:
                self.consecutive_turns += 1
            else:
                self.turn_direction = action
                self.consecutive_turns = 1
        else:
            self.turn_direction = None
            self.consecutive_turns = 0
        observed_turn_count = self.consecutive_turns

        reason = None
        if is_s1 and collision:
            reason = "collision"
        elif is_s1 and action == 1 and displacement is not None and displacement < self.min_forward_displacement:
            reason = "low_gps_displacement"
        elif self.max_consecutive_turns > 0 and observed_turn_count >= self.max_consecutive_turns:
            reason = "repeated_turns"

        if reason is not None:
            self.retry_count += 1
            self.goal_retry_count += 1
            self.consecutive_failures += 1
            self.failed_directions[action] += 1
            if self.current_route_direction is not None:
                self.failed_route_directions[self.current_route_direction] += 1
        elif is_s1 and action == 1:
            self.goal_retry_count = 0
            self.consecutive_failures = 0

        replan_system2 = reason == "repeated_turns" or (
            reason is not None and self.goal_retry_count >= self.system2_retry_limit
        )
        if reason == "repeated_turns":
            self.pending_turn_stall = True
            self.turn_direction = None
            self.consecutive_turns = 0
        return RecoveryDecision(
            cancel_remaining=reason is not None,
            replan_system2=replan_system2,
            reason=reason,
            gps_displacement=displacement,
            retry_count=self.retry_count,
            goal_retry_count=self.goal_retry_count,
            turn_count=observed_turn_count,
        )
