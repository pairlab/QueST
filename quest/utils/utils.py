import copy
import json
import os
import random
from pathlib import Path
import quest.utils.tensor_utils as TensorUtils
import numpy as np
import torch
import torch.nn as nn
import warnings
from natsort import natsorted

def get_experiment_dir(cfg, evaluate=False, allow_overlap=False):
    # if eval_flag:
    #     prefix = "evaluations"
    # else:
    #     prefix = "experiments"
    #     if cfg.pretrain_model_path != "":
    #         prefix += "_finetune"

    prefix = cfg.output_prefix
    if evaluate:
        prefix = os.path.join(prefix, 'evaluate')

    experiment_dir = (
        f"{prefix}/{cfg.task.suite_name}/{cfg.task.benchmark_name}/"
        + f"{cfg.algo.name}/{cfg.exp_name}"
    )
    if cfg.variant_name is not None:
        experiment_dir += f'/{cfg.variant_name}'
    
    if cfg.seed != 10000:
        experiment_dir += f'/{cfg.seed}'

    if cfg.make_unique_experiment_dir:
        # look for the most recent run
        experiment_id = 0
        if os.path.exists(experiment_dir):
            for path in Path(experiment_dir).glob("run_*"):
                if not path.is_dir():
                    continue
                try:
                    folder_id = int(str(path).split("run_")[-1])
                    if folder_id > experiment_id:
                        experiment_id = folder_id
                except BaseException:
                    pass
            experiment_id += 1

        experiment_dir += f"/run_{experiment_id:03d}"
    else:
        experiment_dir += f'/stage_{cfg.stage}'
        
        if not allow_overlap and not cfg.training.resume:
            assert not os.path.exists(experiment_dir), \
                f'cfg.make_unique_experiment_dir=false but {experiment_dir} is already occupied'

    experiment_name = "_".join(experiment_dir.split("/")[len(cfg.output_prefix.split('/')):])
    return experiment_dir, experiment_name

def get_latest_checkpoint(checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    onlyfiles = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
    onlyfiles = natsorted(onlyfiles)
    best_file = onlyfiles[-1]
    return os.path.join(checkpoint_dir, best_file)

def soft_load_state_dict(model, loaded_state_dict):
    # loaded_state_dict['task_encoder.weight'] = loaded_state_dict['task_encodings.weight']
    
    current_model_dict = model.state_dict()
    new_state_dict = {}

    for k in current_model_dict.keys():
        if k in loaded_state_dict:
            v = loaded_state_dict[k]
            if not hasattr(v, 'size') or v.size() == current_model_dict[k].size():
                new_state_dict[k] = v
            else:
                warnings.warn(f'Cannot load checkpoint parameter {k} with shape {loaded_state_dict[k].shape}'
                            f'into model with corresponding parameter shape {current_model_dict[k].shape}. Skipping')
                new_state_dict[k] = current_model_dict[k]
        else:
            new_state_dict[k] = current_model_dict[k]
            warnings.warn(f'Model parameter {k} does not exist in checkpoint. Skipping')
    for k in loaded_state_dict.keys():
        if k not in current_model_dict:
            warnings.warn(f'Loaded checkpoint parameter {k} does not exist in model. Skipping')
    
    model.load_state_dict(new_state_dict)

def map_tensor_to_device(data, device):
    """Move data to the device specified by device."""
    return TensorUtils.map_tensor(
        data, lambda x: safe_device(x, device=device)
    )

def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()

def extract_state_dicts(inp):

    if not (isinstance(inp, dict) or isinstance(inp, list)):
        if hasattr(inp, 'state_dict'):
            return inp.state_dict()
        else:
            return inp
    elif isinstance(inp, list):
        out_list = []
        for value in inp:
            out_list.append(extract_state_dicts(value))
        return out_list
    else:
        out_dict = {}
        for key, value in inp.items():
            out_dict[key] = extract_state_dicts(value)
        return out_dict
        
def save_state(state_dict, path):
    save_dict = extract_state_dicts(state_dict)
    torch.save(save_dict, path)

def load_state(path):
    return torch.load(path)

def torch_save_model(model, optimizer, scheduler, model_path, cfg=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cfg": cfg,
        },
        model_path,
    )

def torch_load_model(model_path):
    checkpoint = torch.load(model_path)
    return checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"], checkpoint["scheduler_state_dict"], checkpoint["cfg"]
