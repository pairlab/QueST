import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import deque

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from quest.algos.baseline_modules.vq_behavior_transformer.utils import MLP
import quest.utils.tensor_utils as TensorUtils


from quest.algos.base import ChunkPolicy

class BehaviorTransformer(ChunkPolicy):
    GOAL_SPEC = Enum("GOAL_SPEC", "concat stack unconditional")

    def __init__(
        self,
        autoencoder,
        policy_prior,
        stage,
        loss_fn,
        offset_loss_multiplier: float = 1.0e3,
        secondary_code_multiplier: float = 0.5,
        frame_stack=10,
        skill_block_size=5,
        sequentially_select=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.autoencoder = autoencoder
        self.policy_prior = policy_prior
        self.stage = stage

        self.frame_stack = frame_stack
        self.skill_block_size = skill_block_size
        self.sequentially_select = sequentially_select
        self._cbet_method = self.GOAL_SPEC.concat
        self._offset_loss_multiplier = offset_loss_multiplier
        self._secondary_code_multiplier = secondary_code_multiplier
        self._criterion = loss_fn

        # For now, we assume the number of clusters is given.
        self._G = self.autoencoder.vqvae_groups  # G(number of groups)
        self._C = self.autoencoder.vqvae_n_embed  # C(number of code integers)
        self._D = self.autoencoder.embedding_dim  # D(embedding dims)

        if self.sequentially_select:
            print("use sequantial prediction for vq dictionary!")
            self._map_to_cbet_preds_bin1 = MLP(
                in_channels=policy_prior.output_dim,
                hidden_channels=[512, 512, self._C],
            )
            self._map_to_cbet_preds_bin2 = MLP(
                in_channels=policy_prior.output_dim + self._C,
                hidden_channels=[512, self._C],
            )
        else:
            self._map_to_cbet_preds_bin = MLP(
                in_channels=policy_prior.output_dim,
                hidden_channels=[1024, 1024, self._G * self._C],
            )
        self._map_to_cbet_preds_offset = MLP(
            in_channels=policy_prior.output_dim,
            hidden_channels=[
                1024,
                1024,
                self._G * self._C * (self.shape_meta.action_dim * self.skill_block_size),
            ],
        )

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.compute_prior_loss(data)
        elif self.stage == 2:
            return self.compute_prior_loss(data)

    def compute_autoencoder_loss(self, data):
        action_input = data["actions"][:, :self.skill_block_size, :]
        pred, total_loss, l1_loss, codebook_loss, pp = self.autoencoder(action_input)
        info = {
            'recon_loss': l1_loss.item(), 
            'codebook_loss': codebook_loss.item(), 
            'pp': pp}
        return total_loss, info
    
    def compute_prior_loss(self, data):
        data = self.preprocess_input(data)

        context = self.get_context(data)
        predicted_action, decoded_action, sampled_centers, logit_info = self._predict(context)
        action_seq = data['actions']
        n, total_w, act_dim = action_seq.shape
        act_w = self.autoencoder.input_dim_h
        obs_w = total_w + 1 - act_w
        output_shape = (n, obs_w, act_w, act_dim)
        output = torch.empty(output_shape, device=action_seq.device)
        for i in range(obs_w):
            output[:, i, :, :] = action_seq[:, i : i + act_w, :]
        action_seq = einops.rearrange(output, "N T W A -> (N T) W A")
        NT = action_seq.shape[0]
        # First, we need to find the closest cluster center for each action.
        state_vq, action_bins = self.autoencoder.get_code(
            action_seq
        )  # action_bins: NT, G

        # Now we can compute the loss.
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)

        offset_loss = torch.nn.L1Loss()(action_seq, predicted_action)

        action_diff = F.mse_loss(
            einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, 0, :
            ],
            einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, 0, :
            ],
        )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        action_diff_tot = F.mse_loss(
            einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, :, :
            ],
            einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, :, :
            ],
        )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        action_diff_mean_res1 = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(decoded_action, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
            )
        ).mean()
        action_diff_mean_res2 = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(
                    predicted_action, "(N T) W A -> N T W A", T=obs_w
                )[:, -1, 0, :]
            )
        ).mean()
        action_diff_max = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(
                    predicted_action, "(N T) W A -> N T W A", T=obs_w
                )[:, -1, 0, :]
            )
        ).max()

        if self.sequentially_select:
            cbet_logits1, gpt_output = logit_info
            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits1[:, :],
                action_bins[:, 0],
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (gpt_output, F.one_hot(action_bins[:, 0], num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits2[:, :],
                action_bins[:, 1],
            )
        else:
            cbet_logits = logit_info
            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 0, :],
                action_bins[:, 0],
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 1, :],
                action_bins[:, 1],
            )
        cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self._secondary_code_multiplier

        equal_total_code_rate = (
            torch.sum(
                (
                    torch.sum((action_bins == sampled_centers).int(), axis=1) == self._G
                ).int()
            )
            / NT
        )
        equal_single_code_rate = torch.sum(
            (action_bins[:, 0] == sampled_centers[:, 0]).int()
        ) / (NT)
        equal_single_code_rate2 = torch.sum(
            (action_bins[:, 1] == sampled_centers[:, 1]).int()
        ) / (NT)

        loss = cbet_loss + self._offset_loss_multiplier * offset_loss
        info = {
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "total_loss": loss.detach().cpu().item(),
            "equal_total_code_rate": equal_total_code_rate.item(),
            "equal_single_code_rate": equal_single_code_rate.item(),
            "equal_single_code_rate2": equal_single_code_rate2.item(),
            "action_diff": action_diff.detach().cpu().item(),
            "action_diff_tot": action_diff_tot.detach().cpu().item(),
            "action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
            "action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
            "action_diff_max": action_diff_max.detach().cpu().item(),
        }
        return loss, info

    def _predict(
        self,
        gpt_input):

        gpt_output = self.policy_prior(gpt_input)

        # there is one task embedding vector in the context so we slice it out here
        gpt_output = gpt_output[:, 1:, :]

        gpt_output = einops.rearrange(gpt_output, "N T (G C) -> (N T) (G C)", G=self._G)

        if self.sequentially_select:
            cbet_logits1 = self._map_to_cbet_preds_bin1(gpt_output)
            cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs1 = torch.softmax(cbet_logits1, dim=-1)
            NT, choices = cbet_probs1.shape
            G = self._G
            sampled_centers1 = einops.rearrange(
                torch.multinomial(cbet_probs1.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (gpt_output, F.one_hot(sampled_centers1, num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_probs2 = torch.softmax(cbet_logits2, dim=-1)
            sampled_centers2 = einops.rearrange(
                torch.multinomial(cbet_probs2.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            sampled_centers = torch.stack(
                (sampled_centers1, sampled_centers2), axis=1
            )  # NT, G
        else:
            cbet_logits = self._map_to_cbet_preds_bin(gpt_output)
            cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
            cbet_logits = einops.rearrange(
                cbet_logits, "(NT) (G C) -> (NT) G C", G=self._G
            )
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs = torch.softmax(cbet_logits, dim=-1)
            NT, G, choices = cbet_probs.shape
            sampled_centers = einops.rearrange(
                torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
                "(NT G) 1 -> NT G",
                NT=NT,
            )

        indices = (
            torch.arange(NT).unsqueeze(1).to(sampled_centers.device),
            torch.arange(self._G).unsqueeze(0).to(sampled_centers.device),
            sampled_centers,
        )
        # Use advanced indexing to sample the values
        sampled_offsets = cbet_offsets[indices]  # NT, G, W, A(?) or NT, G, A

        sampled_offsets = sampled_offsets.sum(dim=1)
        centers = self.autoencoder.draw_code_forward(sampled_centers).view(
            NT, -1, self._D
        )
        return_decoder_input = einops.rearrange(
            centers.clone().detach(), "NT G D -> NT (G D)"
        )
        decoded_action = (
            self.autoencoder.get_action_from_latent(return_decoder_input)
            .clone()
            .detach()
        )  # NT, A
        sampled_offsets = einops.rearrange(sampled_offsets, "NT (W A) -> NT W A", W=self.autoencoder.input_dim_h)
        predicted_action = decoded_action + sampled_offsets

        if self.sequentially_select:
            return predicted_action, decoded_action, sampled_centers, (cbet_logits1, gpt_output)
        return predicted_action, decoded_action, sampled_centers, cbet_logits

    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.autoencoder)
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 2:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
    
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)

        context = self.get_context(data)
        # breakpoint()
        predicted_act, _, _, _ = self._predict(context)

        predicted_act = einops.rearrange(predicted_act, "(N T) W A -> N T W A", T=self.frame_stack)[:, -1, :, :]
        predicted_act = predicted_act.permute(1,0,2)
        return predicted_act.detach().cpu().numpy()

    def get_context(self, data):
        obs_emb = self.obs_encode(data)
        task_emb = self.get_task_emb(data).unsqueeze(1)
        context = torch.cat([task_emb, obs_emb], dim=1)
        return context


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
