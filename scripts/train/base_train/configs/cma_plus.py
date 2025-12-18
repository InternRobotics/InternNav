from internnav.configs.model.cma import cma_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss

cma_plus_exp_cfg = ExpCfg(
    name='cma_plus_train',
    model_name='cma',
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    output_dir='checkpoints/%s/ckpts',
    checkpoint_folder='checkpoints/%s/ckpts',
    tensorboard_dir='checkpoints/%s/tensorboard',
    log_dir='checkpoints/%s/logs',
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_seen'],
        ckpt_to_load='',
        max_steps=195,
        sample=False,
        success_distance=3.0,
        start_eval_epoch=-1,
        step_interval=50,
    ),
    il=IlCfg(
        epochs=55,
        save_interval_epochs=5,
        batch_size=2,
        lr=1e-4,
        num_workers=8,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        use_iw=True,
        inflection_weight_coef=3.2,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='checkpoints/r2r/zero_shot/cma',
        lmdb_map_size=1e12,
        dataset_r2r_root_dir='data/vln_pe/raw_data/r2r',
        dataset_3dgs_root_dir='',
        dataset_grutopia10_root_dir='',
        lmdb_features_dir='r2r',
        lerobot_features_dir='data/vln_pe/traj_data/r2r',
        camera_name='pano_camera_0',
        report_to='wandb',  # wandb, tensorboard, none
        ddp_find_unused_parameters=True,
        filter_failure=FilterFailure(
            use=True,
            min_rgb_nums=15,
        ),
        loss=Loss(
            alpha=0.0001,
            dist_scale=1,
        ),
    ),
    model=cma_cfg,
)
