
# This script is used to train diffusion policy

python train.py --config-name=train_prior.yaml \
    task=libero_90 \
    algo=diffusion_policy \
    exp_name=final \
    variant_name=block_32 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.n_epochs=200 \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    training.auto_continue=true \
    rollout.num_parallel_envs=5 \
    rollout.rollouts_per_env=5 \
    seed=0

# Note1: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
