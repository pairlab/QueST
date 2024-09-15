import torch
import torch.nn as nn
import torch.nn.functional as F
from quest.algos.baseline_modules.diffusion_modules import ConditionalUnet1D
from diffusers.training_utils import EMAModel
from quest.algos.base import ChunkPolicy

class DiffusionPolicy(ChunkPolicy):
    def __init__(
            self, 
            diffusion_model,
            **kwargs
            ):
        super().__init__(**kwargs)
        
        self.diffusion_model = diffusion_model.to(self.device)

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        cond = self.get_cond(data)
        loss = self.diffusion_model(cond, data["actions"])
        info = {
            'loss': loss.item(),
        }
        return loss, info
    
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        cond = self.get_cond(data)
        actions = self.diffusion_model.get_action(cond)
        actions = actions.permute(1,0,2)
        return actions.detach().cpu().numpy()

    def get_cond(self, data):
        obs_emb = self.obs_encode(data)
        obs_emb = obs_emb.reshape(obs_emb.shape[0], -1)
        lang_emb = self.get_task_emb(data)
        cond = torch.cat([obs_emb, lang_emb], dim=-1)
        return cond
    

class DiffusionModel(nn.Module):
    def __init__(self, 
                 noise_scheduler,
                 action_dim,
                 global_cond_dim,
                 diffusion_step_emb_dim,
                 down_dims,
                 ema_power,
                 skill_block_size,
                 diffusion_inf_steps,
                 device):
        super().__init__()
        self.device = device
        net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_emb_dim,
            down_dims=down_dims,
        ).to(self.device)
        self.ema = EMAModel(
            parameters=net.parameters(),
            decay=ema_power)
        self.net = net
        self.noise_scheduler = noise_scheduler
        self.action_dim = action_dim
        self.skill_block_size = skill_block_size
        self.diffusion_inf_steps = diffusion_inf_steps

    def forward(self, cond, actions):
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (cond.shape[0],), device=self.device
        ).long()
        noise = torch.randn(actions.shape, device=self.device)
        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            actions, noise, timesteps)
        # predict the noise residual
        noise_pred = self.net(
            noisy_actions, timesteps, global_cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def get_action(self, cond):
        nets = self.net
        noisy_action = torch.randn(
            (cond.shape[0], self.skill_block_size, self.action_dim), device=self.device)
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(self.diffusion_inf_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets(
                sample=naction, 
                timestep=k,
                global_cond=cond
            )
            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        return naction

    def ema_update(self):
        self.ema.step(self.net.parameters())