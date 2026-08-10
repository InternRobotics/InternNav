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
    ):
        self.min_forward_displacement = float(min_forward_displacement)
        self.system2_retry_limit = int(system2_retry_limit)
        self.max_consecutive_turns = int(max_consecutive_turns)
        self.failed_directions = Counter()
        self.retry_count = 0
        self.goal_retry_count = 0
        self.consecutive_failures = 0
        self.turn_direction = None
        self.consecutive_turns = 0

    def reset(self):
        self.failed_directions.clear()
        self.retry_count = 0
        self.turn_direction = None
        self.consecutive_turns = 0
        self.start_new_goal()

    def start_new_goal(self):
        self.goal_retry_count = 0
        self.consecutive_failures = 0
        self.turn_direction = None
        self.consecutive_turns = 0

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
        elif is_s1 and action == 1:
            self.goal_retry_count = 0
            self.consecutive_failures = 0

        replan_system2 = reason == "repeated_turns" or (
            reason is not None and self.goal_retry_count >= self.system2_retry_limit
        )
        if reason == "repeated_turns":
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
