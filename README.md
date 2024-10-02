# QueST: Self-Supervised Skill Abstractions for Continuous Control

Atharva Mete, Haotian Xue, Albert Wilcox, Yongxin Chen, Animesh Garg

[![Static Badge](https://img.shields.io/badge/Project-Page-green?style=for-the-badge)](https://quest-model.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2407.15840)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Installation**](#installation) | [**Dataset Download**](#dataset-download) | [**Training**](#training) | [**Evaluation**](#evaluating) | [**Project Website**](https://quest-model.github.io/)


<hr style="border: 2px solid gray;"></hr>

## Latest Updates
- [2024-09-25] Accepted at NeurIPS 2024 🎉
- [2024-09-09] Initial release

<hr style="border: 2px solid gray;"></hr>

## Installation

Please run the following commands in the given order to install the dependency for QueST
```
conda create -n quest python=3.10.14
conda activate quest
git clone https://github.com/atharvamete/QueST.git
cd quest
python -m pip install torch==2.2.0 torchvision==0.17.0
python -m pip install -e .
```
Note: Above automatically installs metaworld as python packages

Install LIBERO seperately
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
python -m pip install -e .
```
Note: All LIBERO dependencies are already included in quest/requirements.txt

## Dataset Download
LIBERO: Please download the libero data seperately following their [docs](https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html#datasets).

MetaWorld: We have provided the script we used to collect the data using scripted policies in the MetaWorld package. Please run the following command to collect the data. This uses configs as per [collect_data.yaml](config/collect_data.yaml).
```
python scripts/generate_metaworld_dataset.py
```
We generate 100 demonstrations for each of 45 pretraining tasks and 5 for downstream tasks.

## Training
First set the path to the dataset `data_prefix` and `output_prefix` in [train_base](config/train_base.yaml). `output_prefix` is where all the logs and checkpoints will be stored.

We provide detailed sample commands for training all stages and for all baselines in the [scripts](scripts) directory. For all methods, [autoencoder.sh](scripts/quest/autoencoder.sh) trains the autoencoder (only used in QueST and VQ-BeT), [main.sh](scripts/quest/main.sh) trains the main algorithm (skill-prior incase of QueST), and [finetune.sh](scripts/quest/finetune.sh) finetunes the model on downstream tasks.

Run the following command to train QueST's stage-0 i.e. the autoencoder. (ref: [autoencoder.sh](scripts/quest/autoencoder.sh))
```
python train.py --config-name=train_autoencoder.yaml \
    task=libero_90 \
    algo=quest \
    exp_name=final \
    variant_name=block_32_ds_4 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    seed=0
```
The above command trains the autoencoder on the libero-90 dataset with a block size of 32 and a downsample factor of 4. The run directory will be created at `<output_prefix>/<benchmark_name>/<task>/<algo>/<exp_name>/<variant_name>/<seed>/<stage>`. For above command, it will be `./experiments/libero/libero_90/quest/final/block_32_ds_4/0/stage_0`.

Run the following command to train QueST's stage-1 i.e. the skill-prior. (ref: [main.sh](scripts/quest/main.sh))
```
python train.py --config-name=train_prior.yaml \
    task=libero_90 \
    algo=quest \
    exp_name=final \
    variant_name=block_32_ds_4 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    training.auto_continue=true \
    seed=0
```
Here, training.auto_continue will automatically load the latest checkpoint from the previous training stage.

Run the following command to finetune QueST on a downstream tasks. (ref: [finetune.sh](scripts/quest/finetune.sh))
```
python train.py --config-name=train_fewshot.yaml \
    task=libero_long \
    algo=quest \
    exp_name=final \
    variant_name=block_32_ds_4 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    algo.l1_loss_scale=10 \
    training.auto_continue=true \
    seed=0
```
Here, algo.l1_loss_scale is used to finetune the decoder of the autoencoder while finetuning.

## Evaluating
Run the following command to evaluate the trained model. (ref: [eval.sh](scripts/eval.sh))
```
python evaluate.py \
    task=libero_90 \
    algo=quest \
    exp_name=final \
    variant_name=block_32_ds_4 \
    stage=1 \
    training.use_tqdm=false \
    seed=0
```
This will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage. Else you can specify the checkpoint_path to load a specific checkpoint.

## Citation
If you find this work useful, please consider citing:
```
@misc{mete2024questselfsupervisedskillabstractions,
      title={QueST: Self-Supervised Skill Abstractions for Learning Continuous Control}, 
      author={Atharva Mete and Haotian Xue and Albert Wilcox and Yongxin Chen and Animesh Garg},
      year={2024},
      eprint={2407.15840},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.15840}, 
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<hr style="border: 2px solid gray;"></hr>

## Acknowledgements
1. We would like to thank the authors of [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/) and [MetaWorld](https://meta-world.github.io/) for providing the datasets and environments for our experiments.
2. We would also like to thank the authors of our baselines [VQ-BeT](https://github.com/jayLEE0301/vq_bet_official), [ACT](), and [Diffusion Policy]() for providing the codebase for their methods; and the authors of [Robomimic]() from which we adapted the utility files for our codebase.
