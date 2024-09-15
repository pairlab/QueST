
# This script is used to train the autoencoder of VQ-BeT

python train.py --config-name=train_autoencoder.yaml \
    task=libero_90 \
    algo=bet \
    exp_name=final \
    variant_name=block_5 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    train_dataloader.batch_size=128 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    seed=0