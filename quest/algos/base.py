import torch
import torch.nn as nn
from collections import deque
# from quest.modules.v1 import *
import quest.utils.tensor_utils as TensorUtils
from quest.utils.utils import map_tensor_to_device
import quest.utils.obs_utils as ObsUtils
import einops

from abc import ABC, abstractmethod

class Policy(nn.Module, ABC):
    '''
    Super class with some basic functionality and functions we expect
    from all policy classes in our training loop
    '''

    def __init__(self, 
                 image_encoder_factory,
                 lowdim_encoder_factory,
                 aug_factory,
                 optimizer_factory,
                 scheduler_factory,
                 embed_dim,
                 obs_reduction,
                 shape_meta,
                 device,
                 ):
        super().__init__()

        self.use_augmentation = aug_factory is not None
        self.obs_reduction = obs_reduction
        self.shape_meta = shape_meta
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.device = device
        total_obs_channels = 0

        do_image = image_encoder_factory is not None
        do_lowdim = lowdim_encoder_factory is not None

        # observation encoders
        self.image_encoders = {}
        if do_image and shape_meta['observation']['rgb'] is not None:
            for name, shape in shape_meta["observation"]['rgb'].items():
                shape_in = list(shape)
                encoder = image_encoder_factory(shape_in)
                total_obs_channels += encoder.out_channels
                if obs_reduction == 'stack' and encoder.out_channels != embed_dim:
                    encoder = nn.Sequential(
                        encoder,
                        nn.ReLU(),
                        nn.Linear(encoder.out_channels, embed_dim)
                    )
                self.image_encoders[name] = encoder
            self.image_encoders = nn.ModuleDict(self.image_encoders)
        
        self.lowdim_encoders = {}
        if do_lowdim and shape_meta['observation']['lowdim'] is not None:
            for name, shape in shape_meta['observation']['lowdim'].items():
                encoder = lowdim_encoder_factory(shape)
                total_obs_channels += encoder.out_channels
                if obs_reduction == 'stack' and encoder.out_channels != embed_dim:
                    encoder = nn.Sequential(
                        encoder,
                        nn.ReLU(),
                        nn.Linear(encoder.out_channels, embed_dim)
                    )
                self.lowdim_encoders[name] = encoder
            self.lowdim_encoders = nn.ModuleDict(self.lowdim_encoders)
        
        if obs_reduction == 'cat':
            self.obs_proj = nn.Linear(total_obs_channels, embed_dim)
        else: self.obs_proj = None
        
        if self.use_augmentation:
            self.aug = aug_factory(shape_meta=shape_meta)

        if shape_meta.task.type == "onehot":
            self.task_encoder = nn.Embedding(
                num_embeddings=shape_meta.task.n_tasks,
                embedding_dim=embed_dim
            )
        else:
            self.task_encoder = nn.Linear(shape_meta.task.dim, embed_dim)

        self.device = device

    @abstractmethod
    def compute_loss(self, data):
        raise NotImplementedError('Implement in subclass')

    def get_optimizers(self):
        decay, no_decay = TensorUtils.separate_no_decay(self)
        optimizers = [
            self.optimizer_factory(params=decay),
            self.optimizer_factory(params=no_decay, weight_decay=0.)
        ]
        return optimizers

    def get_schedulers(self, optimizers):
        if self.scheduler_factory is None:
            return []
        else:
            return [self.scheduler_factory(optimizer=optimizer) for optimizer in optimizers]
    
    def preprocess_input(self, data, train_mode=True):
        if train_mode and self.use_augmentation:
            data = self.aug(data)
        for key in self.image_encoders:
            for obs_key in ('obs', 'next_obs'):
                if obs_key in data:
                    x = TensorUtils.to_float(data[obs_key][key])
                    x = x / 255.
                    x = torch.clip(x, 0, 1)
                    data[obs_key][key] = x
        return data

    def obs_encode(self, data, hwc=False, obs_key='obs'):
        ### 1. encode image
        img_encodings, lowdim_encodings = [], []
        for img_name in self.image_encoders.keys():
            x = data[obs_key][img_name]
            if hwc:
                x = einops.rearrange(x, 'B T H W C -> B T C H W')
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name](
                x.reshape(B * T, C, H, W),
                )
            e = e.view(B, T, *e.shape[1:])
            img_encodings.append(e)
        
        # 2. add proprio info
        for lowdim_name in self.lowdim_encoders.keys():
            lowdim_encodings.append(self.lowdim_encoders[lowdim_name](data[obs_key][lowdim_name]))  # add (B, T, H_extra)

        if self.obs_reduction == 'cat':
            encoded = img_encodings + lowdim_encodings
            encoded = torch.cat(encoded, -1)  # (B, T, H_all)
            if self.obs_proj is not None:
                obs_emb = self.obs_proj(encoded)
        elif self.obs_reduction == 'stack':
            encoded = img_encodings + lowdim_encodings
            encoded = torch.stack(encoded, dim=2)
            obs_emb = encoded
        elif self.obs_reduction == 'none':
            return img_encodings, lowdim_encodings
        return obs_emb

    def reset(self):
        return

    def get_task_emb(self, data):
        if "task_emb" in data:
            return self.task_encoder(data["task_emb"])
        else:
            return self.task_encoder(data["task_id"])
    
    def get_action(self, obs, task_id, task_emb=None):
        self.eval()
        for key, value in obs.items():
            if key in self.image_encoders:
                value = ObsUtils.process_frame(value, channel_dim=3)
            obs[key] = torch.tensor(value)
        batch = {}
        batch["obs"] = obs
        if task_emb is not None:
            batch["task_emb"] = task_emb
        else:
            batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
        batch = map_tensor_to_device(batch, self.device)
        with torch.no_grad():
            action = self.sample_actions(batch)
        return action
        
    def preprocess_dataset(self, dataset, use_tqdm=True):
        return

    @abstractmethod
    def sample_actions(self, obs):
        raise NotImplementedError('Implement in subclass')


class ChunkPolicy(Policy):
    '''
    Super class for policies which predict chunks of actions
    '''
    def __init__(self, 
                 action_horizon,
                 **kwargs):
        super().__init__(**kwargs)

        self.action_horizon = action_horizon
        self.action_queue = None


    def reset(self):
        self.action_queue = deque(maxlen=self.action_horizon)
    
    def get_action(self, obs, task_id, task_emb=None):
        assert self.action_queue is not None, "you need to call policy.reset() before getting actions"

        self.eval()
        if len(self.action_queue) == 0:
            for key, value in obs.items():
                if key in self.image_encoders:
                    value = ObsUtils.process_frame(value, channel_dim=3)
                elif key in self.lowdim_encoders:
                    value = TensorUtils.to_float(value) # from double to float
                obs[key] = torch.tensor(value)
            batch = {}
            batch["obs"] = obs
            if task_emb is not None:
                batch["task_emb"] = task_emb
            else:
                batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
            batch = map_tensor_to_device(batch, self.device)
            with torch.no_grad():
                actions = self.sample_actions(batch)
                self.action_queue.extend(actions[:self.action_horizon])
        action = self.action_queue.popleft()
        return action
    
    @abstractmethod
    def sample_actions(self, obs):
        raise NotImplementedError('Implement in subclass')

