from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg, TaskCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8087,
        model_name='dialog',
        ckpt_path='',
        model_settings={
            "mode": "system2",  # inference mode: dual_system or system2
            "dialog_enabled": True,
            "model_path": "projects/VL-LN-Bench/base_model/iign",  # path to model checkpoint
            "append_look_down": False,
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "max_new_tokens": 128,  # maximum number of tokens for generation
        },
    ),
    env=EnvCfg(
        env_type='habitat_vlln',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'habitat_config_path': 'scripts/eval/configs/instance_dialog.yaml',
        },
    ),
    task=TaskCfg(task_name="instance_dialog"),
    eval_type="habitat_dialog",
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/dialog",  # output directory for logs/results
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # task setting
        "eval_split": "easy_same",
        "turn": 5,
        "save_video": True,  # whether to save videos
        # npc setting
        "base_url": 'http://35.220.164.252:3888/v1',
        "model_name": "gpt-4o",
        "openai_api_key": 'internnav/habitat_extensions/vlln/simple_npc/api_key.txt',
        "scene_summary": 'projects/VL-LN-Bench/raw_data/mp3d/scene_summary',
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)
