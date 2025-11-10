from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8087,
        model_name='internvla_n1',
        ckpt_path='',
        model_settings={
            'env_num': 1,
            'sim_num': 1,
            'model_path': "checkpoints/InternVLA-N1",
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            'resize_w': 384,
            'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
            # debug
            'vis_debug': True,  # If vis_debug=True, you can get visualization results
            'vis_debug_path': './logs/test/vis_debug',
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "local_rank": 0,  # node rank
        "output_path": "./logs/habitat/test_refactor_debug",  # output directory for logs/results
        "save_video": False,  # whether to save videos
        "world_size": 1,  # number of distributed processes
        "rank": 0,  # rank of current process
        "gpu": 0,  # gpu id to use
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
        "mode": "dual_system",  # inference mode: dual_system or system2
        "model_path": "checkpoints/InternVLA-N1",  # path to model checkpoint
        "num_future_steps": 4,  # number of future steps for prediction
        "num_frames": 32,  # number of frames used in evaluation
        "num_history": 8,
        "resize_w": 384,  # image resize width
        "resize_h": 384,  # image resize height
        "predict_step_nums": 32,  # number of steps to predict
        "continuous_traj": True,  # whether to use continuous trajectory
        "max_new_tokens": 1024,  # maximum number of tokens for generation
    },
)
