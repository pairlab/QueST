
# This script is used to train stage 1 i.e. the gpt-like transformer prior of Quest

python train.py --config-name=train_prior.yaml \
    task=libero_90 \
    algo=quest \
    exp_name=final \
    variant_name=block_32_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    training.auto_continue=true \
    rollout.num_parallel_envs=5 \
    rollout.rollouts_per_env=5 \
    seed=0

# Note1: training.auto_continue will automatically load the latest checkpoint from the previous training stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.
# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
