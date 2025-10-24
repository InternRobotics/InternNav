from internnav.evaluator.utils.common import check_robot_fall
from internnav.evaluator.utils.stuck_checker import StuckChecker
from internnav.utils.common_log_util import common_logger as log

from ..configs.tasks.vln_eval_task import VLNEvalTaskCfg


def get_action_state(obs, action_name):
    controllers = obs['controllers']
    action_state = controllers[action_name]['finished']
    return action_state


class DoneChecker:
    def __init__(self, offset, robot, config: VLNEvalTaskCfg):
        self._offset = offset
        self.stuck_checker = StuckChecker(offset, robot.articulation)
        self.total_max_step = config.max_step
        self.robot = robot
        self._check_fall_and_stuck_status = config.check_fall_and_stuck
        self._robot_ankle_height = config.robot_ankle_height
        self._fall_height_threshold = config.fall_height_threshold

    def _check_max_steps(self, step):
        if step > self.total_max_step:
            return True, 'exceed_total_max_step'
        return False, ''

    def _check_fall_and_stuck(self, robot_position, robot_rotation, step):
        is_stuck = self.stuck_checker.check_robot_stuck(
            robot_position,
            robot_rotation,
            cur_iter=step,
            max_iter=2500,
            threshold=0.2,
        )
        robot_bottom_z = self.robot.get_ankle_height() - self._robot_ankle_height
        is_fall = check_robot_fall(
            robot_position,
            robot_rotation,
            robot_bottom_z,
            height_threshold=self._fall_height_threshold,
        )

        if is_stuck or is_fall:
            reason = 'fall' if is_fall else 'stuck'
            log.warning(f'Current action has been interrupted by {reason}.')
            return [True], reason
        return [False], ''

    def execute(self, obs, current_action, current_step):
        dones = [False]
        reason = ''
        if current_action == 'stop':
            dones = [True]
            return dones, reason

        over_max_step, desc = self._check_max_steps(current_step)
        if over_max_step:
            dones = [True]
            reason = desc

        if self._check_fall_and_stuck:
            fall_or_stuck, desc = self._check_fall_and_stuck(obs['globalgps'], obs['globalrotation'], current_step)
            if fall_or_stuck[0]:
                dones = [True]
                reason = desc
                log.warning(f'Current action has been interrupted by {reason}.')
                print('robot stuck!')

        return dones, reason
