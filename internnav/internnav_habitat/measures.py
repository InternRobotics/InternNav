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
    """Path Length (PL)
    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
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
    """Oracle Navigation Error (ONE)
    ONE = min(geosdesic_distance(agent_pos, goal)) over all points in the
    agent path.
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
    """Oracle Success Rate (OSR). OSR = I(ONE <= goal_radius)"""

    cls_uuid: str = "oracle_success"

    # def __init__(self, *args: Any, config: Config, **kwargs: Any):
    #     self._config = config
    #     super().__init__()

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
        # self._metric = float(self._metric or d < self._config["success_distance"])
        self._metric = float(self._metric or d < 3.0)


@registry.register_measure
class OracleSPL(Measure):
    """OracleSPL (Oracle Success weighted by Path Length)
    OracleSPL = max(SPL) over all points in the agent path.
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
    """Counts the number of times update_metric() is called. This is equal to
    the number of times that the agent takes an action. STOP counts as an
    action.
    """

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0


# import gzip
# import json
# import pickle
# from dtw import dtw
# from fastdtw import fastdtw
# from habitat.config import Config
# from utils import maps
# from habitat_extensions.task import RxRVLNCEDatasetV1
# from habitat.tasks.nav.nav import DistanceToGoal, Success
# from habitat.tasks.utils import cartesian_to_polar
# from habitat.utils.geometry_utils import quaternion_rotate_vector
# from habitat.utils.visualizations import fog_of_war
# from habitat.utils.visualizations import maps as habitat_maps
# from habitat.core.dataset import Episode
# from habitat.core.embodied_task import Action, EmbodiedTask, Measure
# from habitat.core.logging import logger

# @registry.register_measure
# class WaypointRewardMeasure(Measure):
#     """A reward measure used for training VLN-CE agents via RL."""

#     def __init__(
#         self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
#     ) -> None:
#         self._sim = sim
#         self._slack_reward = config.slack_reward
#         self._use_distance_scaled_slack_reward = (
#             config.use_distance_scaled_slack_reward
#         )
#         self._scale_slack_on_prediction = config.scale_slack_on_prediction
#         self._success_reward = config.success_reward
#         self._distance_scalar = config.distance_scalar
#         self._prev_position = None
#         super().__init__()

#     def reset_metric(
#         self, *args: Any, task: EmbodiedTask, **kwargs: Any
#     ) -> None:
#         task.measurements.check_measure_dependencies(
#             self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
#         )
#         self._previous_distance_to_goal = task.measurements.measures[
#             "distance_to_goal"
#         ].get_metric()
#         self._metric = 0.0
#         self._prev_position = np.take(
#             self._sim.get_agent_state().position, [0, 2]
#         )

#     def _get_scaled_slack_reward(self, action: Action) -> float:
#         if isinstance(action["action"], int):
#             return self._slack_reward

#         if not self._use_distance_scaled_slack_reward:
#             return self._slack_reward

#         agent_pos = np.take(self._sim.get_agent_state().position, [0, 2])
#         slack_distance = (
#             action["action_args"]["r"]
#             if self._scale_slack_on_prediction and action["action"] != "STOP"
#             else np.linalg.norm(self._prev_position - agent_pos)
#         )
#         scaled_slack_reward = self._slack_reward * slack_distance / 0.25
#         self._prev_position = agent_pos
#         return min(self._slack_reward, scaled_slack_reward)

#     def _progress_to_goal(self, task: EmbodiedTask) -> float:
#         distance_to_goal = task.measurements.measures[
#             "distance_to_goal"
#         ].get_metric()
#         distance_to_goal_delta = (
#             self._previous_distance_to_goal - distance_to_goal
#         )
#         if np.isnan(distance_to_goal_delta) or np.isinf(
#             distance_to_goal_delta
#         ):
#             l = self._sim.get_agent_state().position
#             logger.error(
#                 f"\nNaN or inf encountered in distance measure. agent location: {l}",
#             )
#             distance_to_goal_delta = -1.0
#         self._previous_distance_to_goal = distance_to_goal
#         return self._distance_scalar * distance_to_goal_delta

#     def update_metric(
#         self, *args: Any, action: Action, task: EmbodiedTask, **kwargs: Any
#     ) -> None:
#         reward = self._get_scaled_slack_reward(action)
#         reward += self._progress_to_goal(task)
#         reward += (
#             self._success_reward
#             * task.measurements.measures["success"].get_metric()
#         )
#         self._metric = reward

#     @staticmethod
#     def _get_uuid(*args: Any, **kwargs: Any) -> str:
#         return "waypoint_reward_measure"


# @registry.register_measure
# class NDTW(Measure):
#     """NDTW (Normalized Dynamic Time Warping)
#     ref: https://arxiv.org/abs/1907.05446
#     """

#     cls_uuid: str = "ndtw"

#     def __init__(
#         self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
#     ):
#         self._sim = sim
#         self._config = config
#         self.dtw_func = fastdtw if config.FDTW else dtw

#         if "{role}" in config.GT_PATH:
#             self.gt_json = {}
#             for role in RxRVLNCEDatasetV1.annotation_roles:
#                 with gzip.open(
#                     config.GT_PATH.format(split=config.SPLIT, role=role), "rt"
#                 ) as f:
#                     self.gt_json.update(json.load(f))
#         else:
#             with gzip.open(
#                 config.GT_PATH.format(split=config.SPLIT), "rt"
#             ) as f:
#                 self.gt_json = json.load(f)

#         super().__init__()

#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid

#     def reset_metric(self, *args: Any, episode, **kwargs: Any):
#         self.locations = []
#         self.gt_locations = self.gt_json[episode.episode_id]["locations"]
#         self.update_metric()

#     def update_metric(self, *args: Any, **kwargs: Any):
#         current_position = self._sim.get_agent_state().position.tolist()
#         if len(self.locations) == 0:
#             self.locations.append(current_position)
#         else:
#             if current_position == self.locations[-1]:
#                 return
#             self.locations.append(current_position)

#         dtw_distance = self.dtw_func(
#             self.locations, self.gt_locations, dist=euclidean_distance
#         )[0]

#         nDTW = np.exp(
#             -dtw_distance
#             / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE)
#         )
#         self._metric = nDTW


# @registry.register_measure
# class SDTW(Measure):
#     """SDTW (Success Weighted be nDTW)
#     ref: https://arxiv.org/abs/1907.05446
#     """

#     cls_uuid: str = "sdtw"

#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid

#     def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
#         task.measurements.check_measure_dependencies(
#             self.uuid, [NDTW.cls_uuid, Success.cls_uuid]
#         )
#         self.update_metric(task=task)

#     def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
#         ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
#         nDTW = task.measurements.measures[NDTW.cls_uuid].get_metric()
#         self._metric = ep_success * nDTW


# @registry.register_measure
# class TopDownMapVLNCE(Measure):
#     """A top down map that optionally shows VLN-related visual information
#     such as MP3D node locations and MP3D agent traversals.
#     """

#     cls_uuid: str = "top_down_map_vlnce"

#     def __init__(
#         self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
#     ) -> None:
#         self._sim = sim
#         self._config = config
#         self._step_count = None
#         self._map_resolution = config.MAP_RESOLUTION
#         self._previous_xy_location = None
#         self._top_down_map = None
#         self._meters_per_pixel = None
#         self.current_node = ""
#         with open(self._config.GRAPHS_FILE, "rb") as f:
#             self._conn_graphs = pickle.load(f)
#         super().__init__()

#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid

#     def get_original_map(self) -> ndarray:
#         top_down_map = habitat_maps.get_topdown_map_from_sim(
#             self._sim,
#             map_resolution=self._map_resolution,
#             draw_border=self._config.DRAW_BORDER,
#             meters_per_pixel=self._meters_per_pixel,
#         )

#         self._fog_of_war_mask = None
#         if self._config.FOG_OF_WAR.DRAW:
#             self._fog_of_war_mask = np.zeros_like(top_down_map)

#         return top_down_map

#     def reset_metric(
#         self, *args: Any, episode: Episode, **kwargs: Any
#     ) -> None:
#         self._scene_id = episode.scene_id.split("/")[-2]
#         self._step_count = 0
#         self._metric = None
#         self._meters_per_pixel = habitat_maps.calculate_meters_per_pixel(
#             self._map_resolution, self._sim
#         )
#         self._top_down_map = self.get_original_map()
#         agent_position = self._sim.get_agent_state().position
#         scene_id = episode.scene_id.split("/")[-1].split(".")[0]
#         a_x, a_y = habitat_maps.to_grid(
#             agent_position[2],
#             agent_position[0],
#             self._top_down_map.shape[0:2],
#             sim=self._sim,
#         )
#         self._previous_xy_location = (a_y, a_x)

#         if self._config.FOG_OF_WAR.DRAW:
#             self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
#                 self._top_down_map,
#                 self._fog_of_war_mask,
#                 np.array([a_x, a_y]),
#                 self.get_polar_angle(),
#                 fov=self._config.FOG_OF_WAR.FOV,
#                 max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
#                 / habitat_maps.calculate_meters_per_pixel(
#                     self._map_resolution, sim=self._sim
#                 ),
#             )

#         if self._config.DRAW_FIXED_WAYPOINTS:
#             maps.draw_mp3d_nodes(
#                 self._top_down_map,
#                 self._sim,
#                 episode,
#                 self._conn_graphs[scene_id],
#                 self._meters_per_pixel,
#             )

#         if self._config.DRAW_SHORTEST_PATH:
#             shortest_path_points = self._sim.get_straight_shortest_path_points(
#                 agent_position, episode.goals[0].position
#             )
#             maps.draw_straight_shortest_path_points(
#                 self._top_down_map,
#                 self._sim,
#                 self._map_resolution,
#                 shortest_path_points,
#             )

#         if self._config.DRAW_REFERENCE_PATH:
#             maps.draw_reference_path(
#                 self._top_down_map,
#                 self._sim,
#                 episode,
#                 self._map_resolution,
#                 self._meters_per_pixel,
#             )

#         # draw source and target points last to avoid overlap
#         if self._config.DRAW_SOURCE_AND_TARGET:
#             maps.draw_source_and_target(
#                 self._top_down_map,
#                 self._sim,
#                 episode,
#                 self._meters_per_pixel,
#             )

#         # MP3D START NODE
#         self._nearest_node = maps.get_nearest_node(
#             self._conn_graphs[scene_id], np.take(agent_position, (0, 2))
#         )
#         nn_position = self._conn_graphs[self._scene_id].nodes[
#             self._nearest_node
#         ]["position"]
#         self.s_x, self.s_y = habitat_maps.to_grid(
#             nn_position[2],
#             nn_position[0],
#             self._top_down_map.shape[0:2],
#             self._sim,
#         )
#         self.update_metric()

#     def update_metric(self, *args: Any, **kwargs: Any) -> None:
#         self._step_count += 1
#         (
#             house_map,
#             map_agent_pos,
#         ) = self.update_map(self._sim.get_agent_state().position)

#         self._metric = {
#             "map": house_map,
#             "fog_of_war_mask": self._fog_of_war_mask,
#             "agent_map_coord": map_agent_pos,
#             "agent_angle": self.get_polar_angle(),
#             "bounds": {
#                 k: v
#                 for k, v in zip(
#                     ["lower", "upper"],
#                     self._sim.pathfinder.get_bounds(),
#                 )
#             },
#             "meters_per_px": self._meters_per_pixel,
#         }

#     def get_polar_angle(self) -> float:
#         agent_state = self._sim.get_agent_state()
#         # quaternion is in x, y, z, w format
#         ref_rotation = agent_state.rotation

#         heading_vector = quaternion_rotate_vector(
#             ref_rotation.inverse(), np.array([0, 0, -1])
#         )

#         phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
#         z_neg_z_flip = np.pi
#         return np.array(phi) + z_neg_z_flip

#     def update_map(self, agent_position: List[float]) -> None:
#         a_x, a_y = habitat_maps.to_grid(
#             agent_position[2],
#             agent_position[0],
#             self._top_down_map.shape[0:2],
#             self._sim,
#         )
#         # Don't draw over the source point
#         gradient_color = 15 + min(
#             self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
#         )
#         if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
#             maps.drawline(
#                 self._top_down_map,
#                 self._previous_xy_location,
#                 (a_y, a_x),
#                 gradient_color,
#                 thickness=int(
#                     self._map_resolution * 1.4 / maps.MAP_THICKNESS_SCALAR
#                 ),
#                 style="filled",
#             )

#         if self._config.FOG_OF_WAR.DRAW:
#             self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
#                 self._top_down_map,
#                 self._fog_of_war_mask,
#                 np.array([a_x, a_y]),
#                 self.get_polar_angle(),
#                 self._config.FOG_OF_WAR.FOV,
#                 max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
#                 / habitat_maps.calculate_meters_per_pixel(
#                     self._map_resolution, sim=self._sim
#                 ),
#             )

#         point_padding = int(0.2 / self._meters_per_pixel)
#         prev_nearest_node = self._nearest_node
#         self._nearest_node = maps.update_nearest_node(
#             self._conn_graphs[self._scene_id],
#             self._nearest_node,
#             np.take(agent_position, (0, 2)),
#         )
#         if (
#             self._nearest_node != prev_nearest_node
#             and self._config.DRAW_MP3D_AGENT_PATH
#         ):
#             nn_position = self._conn_graphs[self._scene_id].nodes[
#                 self._nearest_node
#             ]["position"]
#             (prev_s_x, prev_s_y) = (self.s_x, self.s_y)
#             self.s_x, self.s_y = habitat_maps.to_grid(
#                 nn_position[2],
#                 nn_position[0],
#                 self._top_down_map.shape[0:2],
#                 self._sim,
#             )
#             self._top_down_map[
#                 self.s_x
#                 - int(2.0 / 3.0 * point_padding) : self.s_x
#                 + int(2.0 / 3.0 * point_padding)
#                 + 1,
#                 self.s_y
#                 - int(2.0 / 3.0 * point_padding) : self.s_y
#                 + int(2.0 / 3.0 * point_padding)
#                 + 1,
#             ] = gradient_color

#             maps.drawline(
#                 self._top_down_map,
#                 (prev_s_y, prev_s_x),
#                 (self.s_y, self.s_x),
#                 gradient_color,
#                 thickness=int(
#                     1.0
#                     / 2.0
#                     * np.round(
#                         self._map_resolution / maps.MAP_THICKNESS_SCALAR
#                     )
#                 ),
#             )

#         self._previous_xy_location = (a_y, a_x)
#         map_agent_pos = (a_x, a_y)
#         return self._top_down_map, map_agent_pos
