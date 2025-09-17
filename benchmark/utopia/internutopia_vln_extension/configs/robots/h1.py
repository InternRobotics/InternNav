from internutopia.macros import gm
from internutopia_extension.configs.controllers import H1MoveBySpeedControllerCfg
from internnav.projects.internutopia_vln_extension.configs.controllers.flash_controller import VlnMoveByFlashControllerCfg

vln_move_by_speed_cfg = H1MoveBySpeedControllerCfg(
    name='vln_move_by_speed',
    type='VlnMoveBySpeedController',
    policy_weights_path='data/Embodiments/vln-pe/h1/policy/move_by_speed/h1_loco_jit_policy.pt',
    joint_names=[
        'left_hip_yaw_joint',
        'right_hip_yaw_joint',
        'torso_joint',
        'left_hip_roll_joint',
        'right_hip_roll_joint',
        'left_shoulder_pitch_joint',
        'right_shoulder_pitch_joint',
        'left_hip_pitch_joint',
        'right_hip_pitch_joint',
        'left_shoulder_roll_joint',
        'right_shoulder_roll_joint',
        'left_knee_joint',
        'right_knee_joint',
        'left_shoulder_yaw_joint',
        'right_shoulder_yaw_joint',
        'left_ankle_joint',
        'right_ankle_joint',
        'left_elbow_joint',
        'right_elbow_joint',
    ],
)
vln_move_by_flash_cfg = VlnMoveByFlashControllerCfg(name='move_by_flash')