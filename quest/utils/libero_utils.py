import copy

# import gym.spaces
# import gym.wrappers
import gymnasium
from collections import OrderedDict, deque
import os
import numpy as np
import quest.utils.file_utils as FileUtils
import quest.utils.obs_utils as ObsUtils
import quest.utils.utils as utils
from PIL import Image
from quest.utils.dataset import SequenceDataset
from torch.utils.data import Dataset
from quest.utils.frame_stack import FrameStackObservationFixed
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
# import gym
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from libero.libero.benchmark import get_benchmark
from transformers import AutoModel, AutoTokenizer, logging
from hydra.utils import to_absolute_path
import time
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
import multiprocessing
import math
import matplotlib.pyplot as plt
import robosuite.utils.transform_utils as T
import h5py
from gymnasium.vector.utils import batch_space
from tqdm import trange
np.set_printoptions(suppress=True)


class LiberoVectorWrapper(gymnasium.Env):
    def __init__(self,
                 env_factory,
                 env_num):
        env_creation, count = False, 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv([env_factory])
                else:
                    env = SubprocVectorEnv([env_factory for _ in range(env_num)])
                env_creation = True
            except Exception as e:
                print(e)
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")
        self._env = env
        self.action_space = batch_space(self._env.action_space[0], env_num)
        self.observation_space = batch_space(self._env.observation_space[0], env_num)

    def reset(self, init_states, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = self.process_obs(obs)
        self._env.set_init_state(init_states)
        return obs, info
    
    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = self._env.step(*args, **kwargs)
        obs = self.process_obs(obs)
        return obs, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def process_obs(self, obs):
        """LIBERO vectorization wrapper does not handle dict obs well"""
        obs_out = {key: [] for key in obs[0]}
        for env_obs in obs:
            for key in obs_out:
                obs_out[key].append(env_obs[key])
        for key in obs_out:
            obs_out[key] = np.array(obs_out[key])
        return obs_out


class LiberoFrameStack(FrameStackObservationFixed):
    def set_init_state(self, *args, **kwargs):
        return self.env.set_init_state(*args, **kwargs)


class LiberoWrapper(gymnasium.Env):
    def __init__(self,
                 task_id,
                 benchmark,
                 shape_meta,
                 obs_key_mapping,
                 img_height=128,
                 img_width=128,
                 cameras=('agentview', 'robot0_eye_in_hand'),
                 device="cuda",):
        self.img_width = img_width
        self.img_height = img_height
        obs_meta = shape_meta['observation']
        self.rgb_outputs = list(obs_meta['rgb'])
        self.lowdim_outputs = list(obs_meta['lowdim'])
        self.cameras = cameras
        self.obs_key_mapping = obs_key_mapping

        self.device = device
        env_args = {
            "bddl_file_name": benchmark.get_task_bddl_file_path(task_id),
            "camera_heights": img_height,
            "camera_widths": img_width,
            'camera_names': cameras
        }

        env = OffScreenRenderEnv(**env_args)
        self.env = env

        obs_space_dict = {}
        for key in self.rgb_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            )
        for key in self.lowdim_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_meta['lowdim'][key],),
                dtype=np.float32
            )
        self.observation_space = gymnasium.spaces.Dict(obs_space_dict)
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.render_out = None

    def reset(self, init_states=None, **kwargs):
        self.env.reset()
        if init_states is not None:
            raw_obs = self.env.set_init_state(init_states)
        # dummy actions [ 7] all zeros for initial physics simulation (as in the original LIBERO code)
        dummy = np.zeros((7,))
        for _ in range(5):
            raw_obs, _, _, _ = self.env.step(dummy)
        return self.make_obs(raw_obs), {}

    def step(self, action):
        raw_obs, reward, truncated, info = self.env.step(action)
        obs = self.make_obs(raw_obs)
        info['success'] = self.env.check_success()
        terminated = info['success']
        return obs, reward, terminated, truncated, info
    
    def set_init_state(self, *args, **kwargs):
        self.env.set_init_state(*args, **kwargs)

    def make_obs(self, raw_obs):
        obs = {}
        self.render_out = raw_obs[f'{self.cameras[0]}_image'][::-1]

        for key in self.rgb_outputs:
            obs[key] = raw_obs[self.obs_key_mapping[key]]

        for key in self.lowdim_outputs:
            obs[key] = raw_obs[self.obs_key_mapping[key]]
        
        return obs
    
    def render(self, *args, **kwargs):
        return self.render_out

def build_dataset(data_prefix,
                  suite_name,
                  benchmark_name, 
                  mode, 
                  seq_len, 
                  frame_stack,
                  shape_meta,
                  n_demos,
                  extra_obs_modality=None,
                  obs_seq_len=1, 
                  load_obs=True,
                  task_embedding_format="clip",
                  ):
    benchmark = get_benchmark(benchmark_name)()
    n_tasks = benchmark.n_tasks
    few_shot_demos = [1, 5, 10, 20, 45] if mode == 'fewshot' else None
    few_shot_demos_list = [f"demo_{i}" for i in few_shot_demos] if few_shot_demos is not None else None
    
    manip_datasets = []
    descriptions = []
    # for key, value in shape_meta
    obs_modality = {
        'rgb': list(shape_meta['observation']['rgb'].keys()),
        'low_dim': list(shape_meta['observation']['lowdim'].keys()),
    }
    if extra_obs_modality is not None:
        for key in extra_obs_modality:
            obs_modality[key] = obs_modality[key] + extra_obs_modality[key]
    # breakpoint()
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for i in trange(n_tasks):
        task_i_dataset = get_dataset(
            dataset_path=os.path.join(
                data_prefix, suite_name, benchmark.get_task_demonstration(i)
            ),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            frame_stack=frame_stack,
            load_obs=load_obs,
            few_demos = few_shot_demos_list,
            n_demos=n_demos,
        )
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
    task_embs = get_task_embs(task_embedding_format, descriptions)
    benchmark.set_task_embs(task_embs)
    datasets = [
        SequenceVLDataset(ds, emb, i) for i,(ds, emb) in enumerate(zip(manip_datasets, task_embs))
    ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")
    return concat_dataset

def get_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    load_obs=True,
    few_demos=None,
    n_demos=None,
    ):
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        few_demos=few_demos,
        n_demos=n_demos,
    )
    return dataset

class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return_dict["task_id"] = self.task_id
        return return_dict

def get_task_embs(task_embedding_format, descriptions):
    logging.set_verbosity_error()
    if task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    return task_embs

