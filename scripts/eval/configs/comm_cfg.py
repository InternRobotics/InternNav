from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg, EvalDatasetCfg, TaskCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8087,
        model_name='cma',
        ckpt_path='checkpoints/r2r/fine_tuned/cma_plus',
        model_settings={},
    ),
    env=EnvCfg['internutopia'],
    task=TaskCfg['vln_pe'],
    dataset=EvalDatasetCfg['mp3d'],
    eval_type='internutopia_vln',
    eval_settings={'save_to_json': False, 'vis_output': True},
)
