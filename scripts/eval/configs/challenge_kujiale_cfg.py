from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8087,
        model_name='rdp',
        ckpt_path='checkpoints/r2r/fine_tuned/rdp',
        model_settings={},
    ),
    env=EnvCfg(
        env_type='vln_pe',
        env_settings={
            'use_fabric': False,
            'headless': True,  # display option: set to False will open isaac-sim interactive window
        },
    ),
    task=TaskCfg(
        task_name='challenge_kujiale_eval',
        task_settings={
            'env_num': 1,
            'use_distributed': False,  # Ray distributed framework
            'proc_num': 8,
        },
        scene=SceneCfg(
            scene_type='kujiale',
            scene_data_dir='interiornav_data/scene_data',
        ),
        robot_name='h1',
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
        camera_resolution=[256, 256],  # (W,H)
        camera_prim_path='torso_link/h1_pano_camera_0',
    ),
    dataset=EvalDatasetCfg(
        dataset_type="kujiale",
        dataset_settings={
            'base_data_dir': 'interiornav_data/raw_data',
            'split_data_types': ['val_unseen', 'val_seen'],
            'filter_stairs': False,
        },
    ),
    eval_settings={'save_to_json': True, 'vis_output': True},  # save result to video under logs/
)
