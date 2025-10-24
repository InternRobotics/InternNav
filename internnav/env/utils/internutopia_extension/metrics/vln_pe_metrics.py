import numpy as np
from internutopia.core.task.metric import BaseMetric

from ..configs.metrics.vln_pe_metrics import VLNPEMetricCfg
from ..configs.tasks.vln_eval_task import VLNEvalTaskCfg

XY_DISTANCE_CLOSE_THRESHHOLD = 1.0


@BaseMetric.register('VLNPEMetric')
class VLNPEMetrics(BaseMetric):
    """
    Calculate the success of this episode
    """

    def __init__(self, config: VLNPEMetricCfg, task_config: VLNEvalTaskCfg):  # runtime get data
        self.config = config
        super().__init__(config, task_config)
        self.shortest_path_length_calc = self.config.shortest_to_goal_distance
        self.success_distance = self.config.success_distance
        self.path_data = task_config.data
        self.shortest_path_length = self.path_data['info']['geodesic_distance']
        self.goal_position = self.path_data['reference_path'][-1]
        self.current_path_length = 0
        self.pred_traj_list = [[]]
        self.metrics = {}
        self.sim_step = 0
        self.fail_reason = []
        self.ne = None
        self.prev_position = None
        self.reset()

    def reset(self):
        pass

    def _calc_ndtw(self):
        dtw_threshold = self.success_distance
        # calculate the cumulative DTW distance between paths
        dtw_distance = 0.0
        trajectory = []
        if len(self.pred_traj_list[0]) > 0:
            trajectory = np.array(self.pred_traj_list[0])[:, :2]  # only take x,y coordinates
            reference_path = np.array(self.path_data['reference_path'])[:, :2]

            for point in trajectory:
                # find the nearest point on the reference path
                min_dist = float('inf')
                for ref_point in reference_path:
                    dist = np.linalg.norm(point - ref_point)
                    min_dist = min(min_dist, dist)
                # accumulate DTW distance, use Gaussian function for normalization
                dtw_distance += np.exp(-(min_dist**2) / (2 * dtw_threshold**2))

        # normalize DTW score
        ndtw = dtw_distance / len(trajectory) if len(trajectory) > 0 else 0.0
        return ndtw

    def update(self, task_obs: dict):  # update related to step
        robot_name = list(task_obs.keys())[0]
        obs = task_obs[robot_name]
        current_position = obs['globalgps']
        self.fail_reason = obs['fail_reason'] if 'fail_reason' in obs else ''

        # update step count
        self.sim_step += 1
        # calculate current path_length
        if self.prev_position is not None:  # initialize, warm_up will change position
            self.current_path_length += np.linalg.norm(
                current_position[:2] - self.prev_position[:2]
            )  # total path length, only xy
        else:
            self.pred_traj_list[0].append(current_position)
        self.prev_position = current_position
        # current
        if obs['finish_action']:
            # add trajectory array
            self.pred_traj_list[0].append(
                current_position
            )  # trajectory array, consider calculation to complete trajectory

            # calculate NE, every round needs
            self.ne = np.linalg.norm(current_position[:2] - self.goal_position[:2])

            # OSR check if it has ever been successful
            self.shortest_path_length_calc = min(self.shortest_path_length_calc, self.ne)

    def calc(self):  # last
        self.metrics['shortest_path_length'] = self.shortest_path_length
        # calculate success distance
        self.metrics['NE'] = self.ne
        self.metrics['success'] = float(self.ne < self.success_distance)

        # OSR check if it has ever been successful
        self.metrics['osr'] = float(self.shortest_path_length_calc < self.success_distance)

        # calculate TL, trajectory total length
        self.metrics['TL'] = self.current_path_length

        # SPL
        self.metrics['spl'] = (
            self.metrics['success']
            * self.shortest_path_length
            / max(self.current_path_length, self.shortest_path_length)
            if self.current_path_length > 0
            else 0
        )

        # calculate NDTW
        self.metrics['ndtw'] = self._calc_ndtw()
        self.metrics['steps'] = self.sim_step

        self.metrics['episode_id'] = self.path_data['episode_id']  # episode ID
        self.metrics['trajectory_id'] = self.path_data['trajectory_id']  # trajectory ID
        self.metrics['fail_reason'] = self.fail_reason
        self.metrics['reference_path'] = self.path_data['reference_path']
        self.metrics['reference_path'] = np.array(self.metrics['reference_path']).tolist()

        return [self.metrics]
