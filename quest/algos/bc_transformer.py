import torch
import torch.nn as nn
import torch.nn.functional as F
import quest.utils.tensor_utils as TensorUtils
from quest.utils.utils import map_tensor_to_device
import quest.utils.obs_utils as ObsUtils
from quest.algos.base import Policy

class BCTransformerPolicy(Policy):
    def __init__(
            self, 
            transformer_model,
            policy_head,
            positional_encoding,
            loss_reduction,
            **kwargs
            ):
        super().__init__(**kwargs) 
        self.temporal_transformer = transformer_model.to(self.device)
        self.policy_head = policy_head.to(self.device)
        self.temporal_position_encoding_fn = positional_encoding.to(self.device)
        self.reduction = loss_reduction

    def temporal_encode(self, x):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def spatial_encode(self, data):
        obs_emb = self.obs_encode(data) # (B, T, num_mod, E)
        text_emb = self.get_task_emb(data)  # (B, E)
        B, T, num_mod, E = obs_emb.shape
        text_emb = text_emb.view(B, 1, 1, -1).expand(-1, T, 1, -1)
        x = torch.cat([text_emb, obs_emb], dim=2)  # (B, T, num_mod+1, E)
        return x

    def forward(self, data):
        x = self.spatial_encode(data)
        x = self.temporal_encode(x)
        dist = self.policy_head(x)
        return dist

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"], self.reduction)
        info = {
            'loss': loss.item(),
        }
        return loss, info
        
    def sample_actions(self, batch):
        batch = self.preprocess_input(batch, train_mode=False)
        x = self.spatial_encode(batch)
        x = self.temporal_encode(x)
        dist = self.policy_head(x[:, -1])
        action = dist.sample().cpu().numpy()
        return action

