
# This script is used to train the ResNet-T model

python train.py --config-name=train_prior.yaml \
    task=libero_90 \
    algo=bc_transformer \
    exp_name=final \
    variant_name=block_10 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    train_dataloader.batch_size=128 \
    make_unique_experiment_dir=false \
    rollout.num_parallel_envs=5 \
    rollout.rollouts_per_env=5 \
    seed=0

# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
