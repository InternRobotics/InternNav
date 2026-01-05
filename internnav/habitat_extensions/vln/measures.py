import gzip
import json
from typing import Any, List, Union

import numpy as np
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import DistanceToGoal
from numpy import ndarray

cv2 = try_cv2_import()


def euclidean_distance(pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class PathLength(Measure):
    """Measure the cumulative path length traveled by the agent by summing the Euclidean distance between consecutive
    agent positions.

    Args:
        sim (Simulator): Simulator instance used to query the agent state and its position at each step.

    Returns:
        float: The total path length accumulated over the current episode.
    """

    cls_uuid: str = "path_length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(current_position, self._previous_position)
        self._previous_position = current_position


@registry.register_measure
class OracleNavigationError(Measure):
    """Track the best (minimum) distance-to-goal achieved at any point along the agent's trajectory during the episode.

    Returns:
        float: The minimum distance-to-goal observed so far in the current episode (initialized to ``inf`` and updated each step).
    """

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_target = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    """Compute oracle success: whether the agent ever gets within a specified goal radius of the target during the
    episode (OSR = I( min_t d_t <= r )).

    Args:
        config (Any): Measure configuration. Typically contains a goal radius (success threshold). Note: the current
            implementation uses a fixed threshold (3.0) instead of reading from ``config``.

    Returns:
        float: 1.0 if the agent is (at the current step, or previously) within the success threshold of the goal,
            otherwise 0.0.
    """

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any):
        print(f"in oracle success init: args = {args}, kwargs = {kwargs}")
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or d < 3.0)


@registry.register_measure
class OracleSPL(Measure):
    """Oracle SPL: track the best (maximum) SPL achieved at any point along the agent's trajectory during the episode.

    Returns:
        float: The maximum SPL observed so far in the current episode (initialized to 0.0 and updated each step).
    """

    cls_uuid: str = "oracle_spl"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, ["spl"])
        self._metric = 0.0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        spl = task.measurements.measures["spl"].get_metric()
        self._metric = max(self._metric, spl)


@registry.register_measure
class StepsTaken(Measure):
    """Count how many actions the agent has taken in the current episode by counting how many times ``update_metric`` is called (including STOP).

    Returns:
        float: The number of steps/actions taken so far in the episode.
    """

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0


from dtw import dtw


@registry.register_measure
class NDTW(Measure):
    r"""'NDTW <https://arxiv.org/abs/1907.05446>'. Computes nDTW between the agent trajectory and a reference
    (ground-truth) path as defined in the paper.

    Args:
        sim (Simulator): The simulator used to query the agent position.
        config (Any): Measure configuration. Note: the current implementation uses hard-coded constants for the
            normalization.

    Returns:
        float: The normalized DTW score in [0, 1], where higher means the executed trajectory is closer to the
            reference path.
    """

    cls_uuid: str = "ndtw"

    def __init__(self, *args: Any, sim: Simulator, config: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.dtw_func = dtw

        # Load ground truth paths from rxr dataset, update this path in habitat config as needed
        gt_json_path = 'data/vln_ce/raw_data/rxr/val_unseen/val_unseen_guide_gt.json.gz'
        with gzip.open(gt_json_path, "rt") as f:
            self.gt_json = json.load(f)

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations = []
        self.gt_locations = self.gt_json[episode.episode_id]["locations"]
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(self.locations, self.gt_locations, dist=euclidean_distance)[0]

        nDTW = np.exp(-dtw_distance / (len(self.gt_locations) * 3.0))  # HARDCODED

        self._metric = nDTW
