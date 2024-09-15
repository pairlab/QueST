# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from quest.algos.baseline_modules.act_utils.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, 
                #  backbones, 
                 transformer, 
                 encoder, 
                 state_dim, 
                 proprio_dim, 
                 num_queries, 
                 shape_meta,
                 encoder_input=('lowdim',)
                 ):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        self.encoder_input = encoder_input
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(proprio_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        
        n_inputs = 0
        obs_meta = shape_meta['observation']
        if 'lowdim' in encoder_input:
            if obs_meta['lowdim'] is not None:
                n_inputs += len(obs_meta['lowdim'])
        if 'perception' in encoder_input:
            if obs_meta['rgb'] is not None:
                n_inputs += len(obs_meta['rgb'])

        self.n_encoder_inputs = n_inputs
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + num_queries + n_inputs, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        n_decoder_inputs = 0
        if obs_meta['lowdim'] is not None:
            n_decoder_inputs += len(obs_meta['lowdim'])
        if obs_meta['rgb'] is not None:
            n_decoder_inputs += len(obs_meta['rgb'])
        self.additional_pos_embed = nn.Embedding(2 + n_decoder_inputs, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, lowdim_encodings, perception_encodings, task_emb, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs = lowdim_encodings.shape[0]
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            
            encoder_input = [cls_embed]
            if 'lowdim' in self.encoder_input:
                encoder_input.append(lowdim_encodings)
            if 'perception' in self.encoder_input:
                encoder_input.append(perception_encodings)
            encoder_input.append(action_embed)
            encoder_input = torch.cat(encoder_input, axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 1 + self.n_encoder_inputs), False).to(lowdim_encodings.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model

            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(lowdim_encodings.device)
            latent_input = self.latent_out_proj(latent_sample)

        task_emb = task_emb.unsqueeze(0)
        latent_input = latent_input.unsqueeze(0)
        lowdim_encodings = lowdim_encodings.permute(1, 0, 2)
        perception_encodings = perception_encodings.permute(1, 0, 2)
        try:
            transformer_input = torch.cat([task_emb, latent_input, lowdim_encodings, perception_encodings], dim=0)
        except RuntimeError:
            raise ValueError(f"something went wrong with the shapes: task_emb: {task_emb.shape}, latent_input: {latent_input.shape}, lowdim_encodings: {lowdim_encodings.shape}, perception_encodings: {perception_encodings.shape}")

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        input_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        hs = self.transformer(transformer_input, None, query_embed, input_pos_embed)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]
    
    def encode_actions(self, actions, qpos):
        pass



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(d_model=256, nheads=8, dim_feedforward=2048, enc_layers=4, pre_norm=False, dropout=0.1):
    activation = "relu"
    encoder_layer = TransformerEncoderLayer(d_model, nheads, dim_feedforward,
                                            dropout, activation, pre_norm)
    encoder_norm = nn.LayerNorm(d_model) if pre_norm else None
    encoder = TransformerEncoder(encoder_layer, enc_layers, encoder_norm)
    return encoder

