import wandb
import numpy as np

class Logger:
    """
    The purpose of this simple logger is to log intermittently and log average values since the last log
    """
    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.data = None

    def update(self, info, step):
        info = flatten_dict(info)
        if self.data is None:
            self.data = {key: [] for key in info}
        
        for key in info:
            self.data[key].append(info[key])
        
        if step % self.log_interval == 0:
            means = {key: np.mean(value) for key, value in self.data.items()}
            self.log(means, step)
            self.data = None

    def log(self, info, step):
        info_flat = flatten_dict(info)
        wandb.log(info_flat, step=step)


def flatten_dict(in_dict):
    """
    The purpose of this is to flatten dictionaries because as of writing wandb handling nested dicts is broken :( 
    https://community.wandb.ai/t/the-wandb-log-function-does-not-treat-nested-dict-as-it-describes-in-the-document/3330
    """

    out_dict = {}
    for key, value in in_dict.items():
        if type(value) is dict:
            for inner_key, inner_value in value.items():
                out_dict[f'{key}/{inner_key}'] = inner_value
        else:
            out_dict[key] = value
    return out_dict