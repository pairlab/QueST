import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import VectorQuantize, FSQ
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer



###############################################################################
#
# Skill-VAE module
#
###############################################################################

def get_fsq_level(codebook_size):
    power = int(np.log2(codebook_size))
    if power == 4: # 16
        fsq_level = [5, 3]
    elif power == 6: # 64
        fsq_level = [8, 8]
    elif power == 8: # 256
        fsq_level = [8, 6, 5]
    elif power == 9: # 512
        fsq_level = [8, 8, 8]
    elif power == 10: # 1024
        fsq_level = [8, 5, 5, 5]
    elif power == 11: # 2048
        fsq_level = [8, 8, 6, 5]
    elif power == 12: # 4096
        fsq_level = [7, 5, 5, 5, 5]
    return fsq_level


class SkillVAE(nn.Module):
    def __init__(self,
                 action_dim,
                 encoder_dim,
                 decoder_dim,
 
                 skill_block_size,
                 downsample_factor, 

                 attn_pdrop,
                 use_causal_encoder,
                 use_causal_decoder,
 
                 encoder_heads,
                 encoder_layers,
                 decoder_heads,
                 decoder_layers,
 
                 vq_type,
                 fsq_level,
                 codebook_dim,
                 codebook_size,
                 ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.skill_block_size = skill_block_size
        self.use_causal_encoder = use_causal_encoder
        self.use_causal_decoder = use_causal_decoder
        self.vq_type = vq_type
        self.fsq_level = fsq_level

        assert int(np.log2(downsample_factor)) == np.log2(downsample_factor), 'downsample_factor must be a power of 2'
        strides = [2] * int(np.log2(downsample_factor)) + [1]
        kernel_sizes = [5] + [3] * int(np.log2(downsample_factor))

        if vq_type == 'vq':
            self.vq = VectorQuantize(dim=encoder_dim, codebook_dim=codebook_dim, codebook_size=codebook_size)
        elif vq_type == 'fsq':
            if fsq_level is None:
                fsq_level = get_fsq_level(codebook_size)
            self.vq = FSQ(dim=encoder_dim, levels=fsq_level)
        else:
            raise NotImplementedError('Unknown vq_type')
        self.action_proj = nn.Linear(action_dim, encoder_dim)
        self.action_head = nn.Linear(decoder_dim, action_dim)
        self.conv_block = ResidualTemporalBlock(
            encoder_dim, encoder_dim, kernel_size=kernel_sizes, 
            stride=strides, causal=use_causal_encoder)

        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, 
                                                   nhead=encoder_heads, 
                                                   dim_feedforward=4*encoder_dim, 
                                                   dropout=attn_pdrop, 
                                                   activation='gelu', 
                                                   batch_first=True, 
                                                   norm_first=True)
        self.encoder =  nn.TransformerEncoder(encoder_layer, 
                                              num_layers=encoder_layers,
                                              enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim,
                                                   nhead=decoder_heads,
                                                   dim_feedforward=4*decoder_dim,
                                                   dropout=attn_pdrop,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.add_positional_emb = Summer(PositionalEncoding1D(encoder_dim))
        self.fixed_positional_emb = PositionalEncoding1D(decoder_dim)
    
    def encode(self, act, obs_emb=None):
        x = self.action_proj(act)
        x = self.conv_block(x)
        B, H, D = x.shape
        
        if obs_emb is not None:
            x = torch.cat([obs_emb, x], dim=1)
        x = self.add_positional_emb(x)

        if self.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.encoder(x, mask=mask, is_causal=True)
        else:
            x = self.encoder(x)

        x = x[:, -H:]

        return x

    def quantize(self, z):
        if self.vq_type == 'vq':
            codes, indices, commitment_loss = self.vq(z)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size, device=z.device)
        else:
            codes, indices = self.vq(z)
            commitment_loss = torch.tensor([0.0], device=z.device)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size, device=z.device)
        ## pp_sample is the average number of unique indices per sequence while pp is for the whole batch
        pp_sample = torch.tensor(np.mean([len(torch.unique(index_seq)) for index_seq in indices])/z.shape[1], device=z.device)
        return codes, indices, pp, pp_sample, commitment_loss

    def decode(self, codes, obs_emb=None):
        x = self.fixed_positional_emb(torch.zeros((codes.shape[0], self.skill_block_size, self.decoder_dim), dtype=codes.dtype, device=codes.device))
        if obs_emb is not None:
            codes = torch.cat([obs_emb, codes], dim=1)
        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.decoder(x, codes, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, codes)
        x = self.action_head(x)
        return x

    def forward(self, act, obs_emb=None):
        z = self.encode(act, obs_emb=obs_emb)
        codes, _, pp, pp_sample, commitment_loss = self.quantize(z)
        x = self.decode(codes, obs_emb=obs_emb)
        return x, pp, pp_sample, commitment_loss, codes

    def get_indices(self, act, obs_emb=None):
        z = self.encode(act, obs_emb=obs_emb)
        _, indices, _, _, _ = self.quantize(z)
        return indices
    
    def decode_actions(self, indices):
        if self.vq_type == 'fsq':
            codes = self.vq.indices_to_codes(indices)
        else:
            codes = self.vq.get_output_from_indices(indices)
        x = self.decode(codes)
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, no_pad=False):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if no_pad:
            self.padding = 0
        else:
            self.padding = dilation*(kernel_size-1)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = (2*self.padding-self.kernel_size)//self.stride + 1
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=4, causal=True, no_pad=False):
        super().__init__()
        if causal:
            conv = CausalConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride, no_pad=no_pad)
        else:
            conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)


# TODO: delete deconv modules for final release version
class CausalDeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super(CausalDeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = self.kernel_size-self.stride
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x

class DeConv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=8, causal=True):
        super().__init__()
        if causal:
            conv = CausalDeConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride)
        else:
            conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride, output_padding=stride-1)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=[5,3], stride=[2,2], n_groups=8, causal=True, residual=False, pooling_layers=[]):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = Conv1dBlock(
                inp_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size[i], 
                stride[i], 
                n_groups=n_groups, 
                causal=causal
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=sum(stride))
        if pooling_layers:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = 0
        for block in self.blocks:
            out = block(out)
            if hasattr(self, 'pooling'):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num += 1
        if hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)
