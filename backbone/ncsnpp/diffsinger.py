# refer to： 
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/model_conformer_naive.py
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/naive_v2_diff.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from ..registry import BackboneRegister

from thop import profile

def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SwiGLU(nn.Module):
    # Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return out * F.silu(gate)

class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0.):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            _dropout
        )

    def forward(self, x):
        return x + self.net(x)


class LYNXNet2(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, expansion_factor=1, kernel_size=31,
                 dropout=0.0, cond_dim=256):
        """
        LYNXNet2(Linear Gated Depthwise Separable Convolution Network Version 2)
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)
        self.conditioner_projection = nn.Linear(cond_dim, num_channels)
        # It may need to be modified at some point to be compatible with the condition cache
        # self.conditioner_projection = nn.Conv1d(hparams['hidden_size'], num_channels, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, in_dims * n_feats)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step=None, cond=None):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """

        if self.n_feats == 1:
            x = spec[:, 0]  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]

        x = self.input_projection(x.transpose(1, 2)) # [B, T, F x M]
        if cond is not None:
            x = x + self.conditioner_projection(cond.transpose(1, 2))

        # It may need to be modified at some point to be compatible with the condition cache
        # x = x + self.conditioner_projection(cond.transpose(1, 2))
        if diffusion_step is not None:
            x = x + self.diffusion_embedding(diffusion_step).unsqueeze(1)

        for layer in self.residual_layers:
            x = layer(x)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=20, num_channels=256, dilation_cycle_length=4):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = Conv1d(in_dims * n_feats, num_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.Mish(),
            nn.Linear(num_channels * 4, num_channels)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=256,
                residual_channels=num_channels,
                dilation=2 ** (i % dilation_cycle_length)
            )
            for i in range(num_layers)
        ])
        self.skip_projection = Conv1d(num_channels, num_channels, 1)
        self.output_projection = Conv1d(num_channels, in_dims * n_feats, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """
        if self.n_feats == 1:
            x = spec.squeeze(1)  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]
        x = self.input_projection(x)  # [B, C, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, M, T]
        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x

@BackboneRegister.register("lynxnet")
class NCSN_LYNXNet2(nn.Module):
    def __init__(self, input_channels, discriminative = False):
        super().__init__()

        self.discriminative = discriminative
        self.core = LYNXNet2(256, 2, num_layers=4, num_channels=256, cond_dim=(input_channels-2)*256)
    
    def forward(self, spec, time=None):
        x = torch.cat([spec[:,0:1,:,:].real, spec[:,0:1,:,:].imag], 1)
        if not self.discriminative:
            cond = torch.cat([spec[:,1:,:,:].real, spec[:,1:,:,:].imag], 1)
            cond = cond.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]
        else:
            cond = None
        out = self.core(x, time, cond)
        spec_enh = out.permute(0,2,3,1).unsqueeze(1)  # (B,F,T,2)
        
        spec_enh = spec_enh.contiguous().to(torch.float32)
        return torch.view_as_complex(spec_enh)


if __name__ == '__main__':

    # LynxNet
    lynxNet = LYNXNet2(256, 2, num_layers=4, num_channels=256, )
    # WaveNet
    waveNet = WaveNet(256, 2, num_layers=4, dilation_cycle_length=1)
    
    # input
    x = torch.randn(1, 2, 256, 256)    #  B 1 D T
    diffusion_step = torch.tensor([1])
    cond = torch.randn(1, 256, 256)
    
    flops, params = profile(lynxNet, inputs=(x, diffusion_step, cond), verbose=False)
    print(f"| LynxNet: 单次前向 FLOPs: {flops / 1e9:.2f} GFLOPs | MACs: {flops / 1000000000.0 :.2f} G | Params: {params / 1000000.0:.2f} M")
    
    flops, params = profile(waveNet, inputs=(x, diffusion_step, cond), verbose=False)
    print(f"| WaveNet: 单次前向 FLOPs: {flops / 1e9:.2f} GFLOPs | MACs: {flops / 1000000000.0 :.2f} G | Params: {params / 1000000.0:.2f} M")
    
    out = lynxNet(x, diffusion_step, cond)
    print(x.shape, out.shape)

    # LynxNet
    x = torch.randn(1, 2, 256, 256)    #  B 1 D T
    x_complex = torch.complex(x, x)  # 实部=虚部
    nc = NCSN_LYNXNet2(4)
    out = nc(x_complex, diffusion_step)
    flops, params = profile(nc, inputs=(x_complex, diffusion_step), verbose=False)
    print(f"| NCSN_LYNXNet2: 单次前向 FLOPs: {flops / 1e9:.2f} GFLOPs | MACs: {flops / 1000000000.0 :.2f} G | Params: {params / 1000000.0:.2f} M")
    
