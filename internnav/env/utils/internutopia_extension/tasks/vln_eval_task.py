from internutopia.core.task import BaseTask

from internnav.evaluator.utils.common import set_seed

from ..configs.tasks.vln_eval_task import VLNEvalTaskCfg
from .utils import DoneChecker


@BaseTask.register('VLNEvalTask')
class VLNEvalTask(BaseTask):
    def __init__(self, config: VLNEvalTaskCfg, scene):
        set_seed(0)
        super().__init__(config, scene)
        self.step_count = 0
        self.data = config.data
        self.warm_up_step = config.warm_up_step
        self.done_checker: DoneChecker = None
        self._done = None
        self.config = config

    def _get_robot_poses_without_offset(self):
        pre_position, pre_rotation = self.robot.articulation.get_world_pose()
        positions = pre_position - self.env_offset
        orientations = pre_rotation
        return positions, orientations

    def _set_warmup_step(self, step):
        self.warm_up_step = step

    def create_light(self):
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        stage.RemovePrim('/World/distant_light')
        distant_light = UsdLux.DistantLight.Define(stage, self.root_path + '/distant_light')
        distant_light.CreateIntensityAttr(1000)
        distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        self.distant_light = distant_light
        self.distant_light_position = UsdGeom.Xformable(self.distant_light).AddTranslateOp()
        self.distant_light_position.Set(Gf.Vec3f(*self.env_offset))

        stage.RemovePrim('/World/up_disk_light')
        up_disk_light = UsdLux.DiskLight.Define(stage, self.root_path + '/up_disk_light')
        up_disk_light.CreateIntensityAttr(5000)
        up_disk_light.CreateRadiusAttr(50.0)
        up_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(up_disk_light).AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))

        self.up_disk_light = up_disk_light
        self.up_disk_light_position = UsdGeom.Xformable(self.up_disk_light).AddTranslateOp()
        self.up_disk_light_position.Set(Gf.Vec3f(*self.env_offset))

        stage.RemovePrim('/World/down_disk_light')
        down_disk_light = UsdLux.DiskLight.Define(stage, self.root_path + '/down_disk_light')
        down_disk_light.CreateIntensityAttr(5000)
        down_disk_light.CreateRadiusAttr(50.0)
        down_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        self.down_disk_light = down_disk_light
        self.down_disk_light_position = UsdGeom.Xformable(self.down_disk_light).AddTranslateOp()
        self.down_disk_light_position.Set(Gf.Vec3f(*self.env_offset))

    def reset_light_position(self, position):
        from pxr import Gf

        raise_light = 1
        new_position = position + self.env_offset
        self.up_disk_light_position.Set(
            Gf.Vec3f(
                new_position[0],
                -new_position[1],
                -new_position[2] - raise_light,
            )
        )
        self.down_disk_light_position.Set(Gf.Vec3f(new_position[0], new_position[1], new_position[2] + raise_light))

    def load(self):
        super().load()
        self.robot_name = list(self.robots.keys())[0]
        self.create_light()
        self.done_checker = DoneChecker(
            self.env_offset,
            self.robots[list(self.robots.keys())[0]],
            self.config,
        )

    def post_reset(self) -> None:
        self.steps = 0
        for robot in self.robots.values():
            robot.post_reset()
        self.reset_light_position(self.data['start_position'])
        self.robot = self.robots[list(self.robots.keys())[0]]
        self.articulation = self.robot.articulation

    def is_done(self) -> bool:
        return self._done if self._done is not None else False

    def get_rgb_depth(self):
        obs = {}
        if 'pano_camera_0' in self.robot.sensors:
            camera = self.robot.sensors['pano_camera_0']
            import omni.replicator.core as rep

            if self.env_id == 0:
                rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
            cur_obs = camera.get_data()
            rgb_info = cur_obs['rgba'][..., :3]

            import numpy as np

            from internnav.evaluator.utils.common import norm_depth

            depth_info = norm_depth(cur_obs['depth'])
            obs['depth'] = depth_info[..., np.newaxis]
            obs['rgb'] = rgb_info

        if 'topdown_camera_500' in self.robot.sensors:
            topdown_global_map_camera = self.robot.sensors['topdown_camera_500']
            cur_obs = topdown_global_map_camera.get_data()
            obs['topdown_rgb'] = cur_obs['rgba'][..., :3]
            obs['topdown_depth'] = norm_depth(cur_obs['depth'])
        return obs

    def update_metrics(self, obs):
        for metric in self.metrics.values():
            metric.update(obs)

    def get_current_action(self):
        return self.robot.current_action

    def get_observations(self):

        obs = {}
        obs['finish_action'] = False
        (
            obs['globalgps'],
            obs['globalrotation'],
        ) = self._get_robot_poses_without_offset()
        if self._done:
            obs['finish_action'] = True
            obs['metrics'] = self.calculate_metrics()

        current_action = self.robot.current_action
        if current_action is None:
            # reset
            return {self.robot_name: obs}
        action_name = list(current_action.keys())[0]

        # add step
        self.step_count = self.step_count + 1
        assert action_name in [
            'stand_still',
            'move_by_discrete',
            'vln_move_by_speed',
            'vln_dp_move_by_speed',
            'move_by_flash',
            'stop',
        ], f"Got invalid action name {action_name}!!!"
        if action_name == 'stand_still':
            if self.warm_up_step > 1:
                self.step_count -= 1
                self.warm_up_step -= 1
                self.robot.current_action = None
                return {self.robot_name: obs}
            else:
                obs.update(self.get_rgb_depth())
                if (not self.config.robot_flash) and (not self.config.one_step_stand_still):
                    self.warm_up_step = 50  # without this, possible issues: delay by get_rgb; break warm up

        elif action_name == 'move_by_discrete':
            move_by_discrete_obs = self.robot.controllers['move_by_discrete'].get_obs()
            if not move_by_discrete_obs['finished']:
                self.robot.current_action = None
                return {self.robot_name: obs}
            obs.update(self.get_rgb_depth())

        elif action_name == 'vln_move_by_speed':
            move_by_speed_obs = self.robot.controllers['vln_move_by_speed'].get_obs()
            if not move_by_speed_obs['finished']:
                return {self.robot_name: obs}  # not finish
            obs.update(self.get_rgb_depth())

        elif action_name == 'vln_dp_move_by_speed':
            move_by_speed_obs = self.robot.controllers['vln_dp_move_by_speed'].get_obs()
            if not move_by_speed_obs['finished']:
                return {self.robot_name: obs}  # not finish
            obs.update(self.get_rgb_depth())

        elif action_name == 'move_by_flash':
            obs.update(self.get_rgb_depth())

        obs['finish_action'] = True
        self.robot.current_action = None
        # update when stop
        dones, reason = self.done_checker.execute(obs, action_name, self.step_count)
        self._done = dones[0]
        if self._done:
            self.update_metrics({self.robot_name: obs})
            obs['metrics'] = self.calculate_metrics()

            if action_name == 'stop':
                if obs['metrics'][list(obs['metrics'].keys())[0]][0]['success'] > 0:
                    reason = 'success'
                else:
                    reason = 'not_reach_goal'
            obs['metrics'][list(obs['metrics'].keys())[0]][0]['fail_reason'] = reason
            for metric in self.metrics.values():
                metric.fail_reason = reason

        # calculate metrics
        obs['fail_reason'] = reason
        obs['instruction'] = self.data['instruction']['instruction_text']
        obs['instruction_tokens'] = self.data['instruction']['instruction_tokens']

        obs = {self.robot_name: obs}
        return obs
