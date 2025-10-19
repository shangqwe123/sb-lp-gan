import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import sys
sys.path.append('/gpfs/home/shangzengqiang/shangzengqiang/liubiao/')
from AP_BWE.utils import init_weights, get_padding
import numpy as np
from typing import Tuple, List
import json
from AP_BWE.env import AttrDict, build_env
from backbone.ncsnpp.ncsnpp_utils import layerspp
from backbone.registry import BackboneRegister
LRELU_SLOPE = 0.1

class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim=128):
        super().__init__()
        # If passed a boolean True/False, default to 128 as cond dim
        if isinstance(cond_dim, bool):
            cond_dim = 128
        self.base_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2)
        )

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor):
        # x: (B, T, C), cond_embedding: (B, E)
        x = self.base_norm(x)
        if cond_embedding is None:
            return x
        scale, shift = self.cond_mlp(cond_embedding).chunk(2, dim=-1)  # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        layer_scale_init_value= None,
        adanorm_num_embeddings = None,
    ):
        super().__init__()
        # self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=0, groups=dim)  # 去掉 padding
        self.adanorm = adanorm_num_embeddings is not None
        
        if self.adanorm:
            self.norm = AdaLayerNorm(dim, adanorm_num_embeddings)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim*3)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim*3, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, cond_embedding_id = None) :
        residual = x
        x = F.pad(x, (self.dwconv.kernel_size[0] - 1, 0))  # 左 pad,保持因果，不需要左右都pad
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

@BackboneRegister.register("apnet_bwe_model")
class APNet_BWE_Model(torch.nn.Module):
    def __init__(self, input_channels=2, spatial_channels=1, image_size=256, discriminative=False):
        super(APNet_BWE_Model, self).__init__()
        # with open('/gpfs/home/shangzengqiang/shangzengqiang/liubiao/AP_BWE/configs/config_2kto16k_new.json') as f:
        #     data = f.read()
        # json_config = json.loads(data)
        # h = AttrDict(json_config)
        with open('/gpfs/home/shangzengqiang/shangzengqiang/exp_oppo/MISB-main/config_2kto16k_new.json') as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        self.h = h
        self.adanorm_num_embeddings = 128
        layer_scale_init_value =  1 / h.ConvNeXt_layers

        # self.conv_pre_mag = nn.Conv1d(h.n_fft//2+1, h.ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.conv_pre_mag = nn.Conv1d((h.n_fft//2+1)*(input_channels//2), h.ConvNeXt_channels, 7, 1, padding=0)
        self.norm_pre_mag = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        # self.conv_pre_pha = nn.Conv1d(h.n_fft//2+1, h.ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.conv_pre_pha = nn.Conv1d((h.n_fft//2+1)*(input_channels//2), h.ConvNeXt_channels, 7, 1, padding=0)
        self.norm_pre_pha = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.discriminative = discriminative

        self.convnext_mag = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=h.ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings*2 if not self.discriminative else None,
                )
                for _ in range(h.ConvNeXt_layers)
            ]
        )

        self.convnext_pha = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=h.ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings*2 if not self.discriminative else None,
                )
                for _ in range(h.ConvNeXt_layers)
            ]
        )

        self.norm_post_mag = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.apply(self._init_weights)
        self.linear_post_mag = nn.Linear(h.ConvNeXt_channels, h.n_fft//2+1)
        self.linear_post_pha_r = nn.Linear(h.ConvNeXt_channels, h.n_fft//2+1)
        self.linear_post_pha_i = nn.Linear(h.ConvNeXt_channels, h.n_fft//2+1)
        
        self.time_embedding = layerspp.GaussianFourierProjection(
                embedding_size=self.adanorm_num_embeddings, scale=16
            )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, time_cond=None):
    # def forward(self, mag_nb):
    #     input_shape = (257, 63)
    #     pha_nb = torch.randn(1, *input_shape)
        # x_mag = self.conv_pre_mag(mag_nb)
        # x_pha = self.conv_pre_pha(pha_nb)
            # Gaussian Fourier features embeddings.
        temb = self.time_embedding(torch.log(time_cond)) if not self.discriminative else None

        mag_nb = x.real.reshape(x.shape[0], -1, x.shape[-1])
        pha_nb = x.imag.reshape(x.shape[0], -1, x.shape[-1])

        x_mag = F.pad(mag_nb, (6, 0))    ## 只需要左pad，不进行左右pad，保持因果
        x_mag = self.conv_pre_mag(x_mag)
        x_pha = F.pad(pha_nb, (6, 0))
        x_pha = self.conv_pre_pha(x_pha)
        x_mag = self.norm_pre_mag(x_mag.transpose(1, 2)).transpose(1, 2)
        x_pha = self.norm_pre_pha(x_pha.transpose(1, 2)).transpose(1, 2)

        for conv_block_mag, conv_block_pha in zip(self.convnext_mag, self.convnext_pha):
            x_mag = x_mag + x_pha
            x_pha = x_pha + x_mag
            x_mag = conv_block_mag(x_mag, cond_embedding_id=temb)
            x_pha = conv_block_pha(x_pha, cond_embedding_id=temb)

        x_mag = self.norm_post_mag(x_mag.transpose(1, 2))
        mag_wb = self.linear_post_mag(x_mag).transpose(1, 2)
        
        x_pha = self.norm_post_pha(x_pha.transpose(1, 2))
        x_pha_r = self.linear_post_pha_r(x_pha)
        x_pha_i = self.linear_post_pha_i(x_pha)
        pha_wb = torch.atan2(x_pha_i, x_pha_r).transpose(1, 2)

        com_wb = torch.stack((torch.exp(mag_wb)*torch.cos(pha_wb), 
                           torch.exp(mag_wb)*torch.sin(pha_wb)), dim=-1)
        
        # 根据mag和pha变成复数输出
        # mag_wb: (B, F, T), pha_wb: (B, F, T)
        # 输出复数：(B, F, T)
        complex_wb = torch.exp(mag_wb) * torch.exp(1j * pha_wb)
        #return mag_wb, pha_wb, com_wb
        #print(complex_wb.shape)
        return complex_wb.unsqueeze(1)



class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i,l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            if i > 0:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiResolutionAmplitudeDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: Tuple[Tuple[int, int, int]] = ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)),
        num_embeddings: int = None,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorAR(resolution=r, num_embeddings=num_embeddings) for r in resolutions]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorAR(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int, int],
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: int = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(in_channels, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=channels)
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(
        self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x=x.squeeze(1)
        
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        amplitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=None,  # interestingly rectangular window kind of works here
            center=True,
            return_complex=True,
        ).abs()

        return amplitude_spectrogram


class MultiResolutionPhaseDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: Tuple[Tuple[int, int, int]] = ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)),
        num_embeddings: int = None,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorPR(resolution=r, num_embeddings=num_embeddings) for r in resolutions]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorPR(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int, int],
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: int = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(in_channels, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=channels)
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(
        self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x=x.squeeze(1)
        
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        phase_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=None,  # interestingly rectangular window kind of works here
            center=True,
            return_complex=True,
        ).angle()

        return phase_spectrogram


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

    return loss, gen_losses


def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def stft_mag(audio, n_fft=2048, hop_length=512):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length, window=hann_window, return_complex=True)
    stft_mag = torch.abs(stft_spec)
    return(stft_mag)

def cal_snr(pred, target):
    snr = (20 * torch.log10(torch.norm(target, dim=-1) / torch.norm(pred - target, dim=-1).clamp(min=1e-8))).mean()
    return snr

def cal_lsd(pred, target):
    sp = torch.log10(stft_mag(pred).square().clamp(1e-8))
    st = torch.log10(stft_mag(target).square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()



if __name__ == "__main__":
    model = APNet_BWE_Model().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print(flops, params)
