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
        model_name='cma',
        ckpt_path='checkpoints/r2r/fine_tuned/cma',
        model_settings={},
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='cma_kujiale_eval',
        task_settings={
            'env_num': 2,
            'use_distributed': True,
            'proc_num': 4,
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
)
