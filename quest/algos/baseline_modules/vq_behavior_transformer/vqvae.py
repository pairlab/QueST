import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
import numpy as np
import einops
from quest.algos.baseline_modules.vector_quantize_pytorch_bet.residual_vq import ResidualVQ

class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=1,
        last_activation=None,
    ):
        super(EncoderMLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state


class VqVae(nn.Module):
    def __init__(
        self,
        obs_dim=60,
        input_dim_h=1,  # length of action chunk
        input_dim_w=7,  # action dim
        n_latent_dims=512,
        vqvae_n_embed=16,
        vqvae_groups=2,
        hidden_dim=128,
        num_layers=1,
        device="cuda",
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
    ):
        super().__init__()
        self.n_latent_dims = n_latent_dims
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.rep_dim = self.n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vqvae_lr = 1e-3
        self.vqvae_groups = vqvae_groups
        self.device = device
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

        self.vq_layer = ResidualVQ(
            dim=self.n_latent_dims,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
        ).to(self.device)
        self.embedding_dim = self.n_latent_dims

        self.vq_layer.device = device

        if self.input_dim_h == 1:
            self.encoder = EncoderMLP(
                input_dim=input_dim_w, hidden_dim=self.hidden_dim, layer_num=self.num_layers, output_dim=n_latent_dims
            ).to(self.device)
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims, hidden_dim=self.hidden_dim, layer_num=self.num_layers, output_dim=input_dim_w
            ).to(self.device)
        else:
            self.encoder = EncoderMLP(
                input_dim=input_dim_w * self.input_dim_h, hidden_dim=self.hidden_dim, layer_num=self.num_layers, output_dim=n_latent_dims
            ).to(self.device)
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims, hidden_dim=self.hidden_dim, layer_num=self.num_layers, output_dim=input_dim_w * self.input_dim_h
            ).to(self.device)

        # params = (
        #     list(self.encoder.parameters())
        #     + list(self.decoder.parameters())
        #     + list(self.vq_layer.parameters())
        # )
        # self.vqvae_optimizer = torch.optim.Adam(
        #     params, lr=self.vqvae_lr, weight_decay=0.0001
        # )

        # if load_dir is not None:
        #     try:
        #         state_dict = torch.load(load_dir)
        #     except RuntimeError:
        #         state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
        #     self.load_state_dict(state_dict)

        # if eval:
        #     self.vq_layer.eval()
        # else:
        #     self.vq_layer.train()

    def draw_logits_forward(self, encoding_logits):
        z_embed = self.vq_layer.draw_logits_forward(encoding_logits)
        return z_embed

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent):
        output = self.decoder(latent) * self.act_scale
        if self.input_dim_h == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if self.input_dim_h == 1:
            state = state.squeeze(-2)  # state.squeeze(-1)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def get_code(self, state, required_recon=False):
        state = state / self.act_scale
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
            if required_recon:
                recon_state = self.decoder(state_vq) * self.act_scale
                recon_state_ae = self.decoder(state_rep) * self.act_scale
                if self.input_dim_h == 1:
                    return state_vq, vq_code, recon_state, recon_state_ae
                else:
                    return (
                        state_vq,
                        vq_code,
                        torch.swapaxes(recon_state, -2, -1),
                        torch.swapaxes(recon_state_ae, -2, -1),
                    )
            else:
                # econ_from_code = self.draw_code_forward(vq_code)
                return state_vq, vq_code

    def forward(self, state):
        state = state / self.act_scale
        state = self.preprocess(state)
        state_rep = self.encoder(state)
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        vq_loss_state = torch.sum(vq_loss_state)

        dec_out = self.decoder(state_vq)
        encoder_loss = (state - dec_out).abs().mean()
        rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
        pp = len(torch.unique(vq_code))/1024

        return dec_out, rep_loss, encoder_loss.clone().detach(), vq_loss_state.clone().detach(), pp
        

    # def state_dict(self):
    #     return {
    #         "encoder": self.encoder.state_dict(),
    #         "decoder": self.decoder.state_dict(),
    #         "vq_embedding": self.vq_layer.state_dict(),
    #     }

    # def load_state_dict(self, state_dict):
    #     self.encoder.load_state_dict(state_dict["encoder"])
    #     self.decoder.load_state_dict(state_dict["decoder"])
    #     self.vq_layer.load_state_dict(state_dict["vq_embedding"])
    #     self.vq_layer.eval()


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


# def var(tensor):
#     return tensor.to(device)


def get_tensor(z, device):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return torch.FloatTensor(z.copy()).to(device).unsqueeze(0)
        # return torch.from_numpy(z.copy()).float().to(device).unsqueeze(0)
    else:
        return torch.FloatTensor(z.copy()).to(device)
        # return torch.from_numpy(z.copy()).float().to(device)