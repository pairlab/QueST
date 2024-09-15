import numpy as np
import h5py
from tqdm import tqdm
import json

import quest.utils.metaworld_utils as mu
import os
import hydra
from hydra.utils import instantiate
import quest.utils.utils as utils
from moviepy.editor import ImageSequenceClip

@hydra.main(config_path="../config", 
            config_name='collect_data', 
            version_base=None)
def main(cfg):
    env_runner = instantiate(cfg.task.env_runner)

    data_dir = os.path.join(
                cfg.data_prefix, 
                cfg.task.suite_name,
                cfg.task.benchmark_name,
                cfg.task.mode
                # f"{task_names[i]}.hdf5"
            )
    os.makedirs(data_dir, exist_ok=True)
    experiment_dir, _ = utils.get_experiment_dir(cfg)
    
    success_rates, returns = {}, {}
    expert = mu.get_expert()

    def noisy_expert(obs, task_id):
        expert_action = expert(obs, task_id)
        action = np.random.normal(expert_action, cfg.task.demo_noise)
        action = np.clip(action, -1, 1)
        return action

    for env_name in mu.get_env_names(cfg.task.benchmark_name, cfg.task.mode):
        file_path = os.path.join(data_dir, f"{env_name}.hdf5")
        if os.path.exists(file_path):
            print(f'{file_path} already exists. Skipping')
            continue
        video_dir = os.path.join(experiment_dir, env_name)
        os.makedirs(video_dir)
        init_hdf5(file_path, env_name)
        
        completed = total_return = 0
        rollouts = env_runner.run_policy_in_env(env_name, noisy_expert)
        for i, (success, ep_return, episode) in tqdm(enumerate(rollouts), total=cfg.rollout.rollouts_per_env):

            completed += success
            total_return += ep_return

            save_path = os.path.join(video_dir, f'trial_{i}.mp4')
            clip = ImageSequenceClip(list(episode['corner_rgb']), fps=24)
            clip.write_videofile(save_path, fps=24, verbose=False, logger=None)
            dump_demo(episode, file_path, i)
        success_rate = completed / (i + 1)
        success_rates[env_name] = success_rate
        returns[env_name] = total_return / (i + 1)
        print(env_name, success_rate)

    with open(os.path.join(data_dir, 'success_rates.json'), 'w') as f:
        json.dump(success_rates, f)
    with open(os.path.join(data_dir, 'returns.json'), 'w') as f:
        json.dump(returns, f)


def init_hdf5(file_path, env_name):
    with h5py.File(file_path, 'a') as f:
        group_data = f.create_group('data')
        group_data.attrs['total'] = 0
        group_data.attrs['env_args'] = json.dumps({
            'env_name': env_name, 'env_type': 2, 
            'env_kwargs':{'render_mode':'rgb_array', 'camera_name':'corner2'}
            })

def dump_demo(demo, file_path, demo_i):
    with h5py.File(file_path, 'a') as f:
        group_data = f['data']
        group = group_data.create_group(f'demo_{demo_i}')

        demo_length = demo['actions'].shape[0]
        group_data.attrs['total'] = group_data.attrs['total'] + demo_length
        group.attrs['num_samples'] = demo_length
        non_obs_keys = ('actions', 'terminated', 'truncated', 'reward', 'success')
        group.create_dataset('states', data=())
        for key in demo:
            if key in non_obs_keys:
                continue
            group.create_dataset(f'obs/{key}', data=demo[key])
        for key in non_obs_keys:
            group.create_dataset(key, data=demo[key])


if __name__ == '__main__':
    main()