
python evaluate.py \
    task=libero_90 \
    algo=quest \
    exp_name=final \
    variant_name=block_32_ds_4 \
    stage=1 \
    training.use_tqdm=false \
    seed=0

# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.