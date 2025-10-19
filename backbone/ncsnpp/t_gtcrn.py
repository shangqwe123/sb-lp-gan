import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from ..registry import BackboneRegister

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=510, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1 - erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class TimeAwareConvBlock(nn.Module):
    """ConvBlock with time embedding injection"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 groups=1, use_deconv=False, is_last=False, time_dim=32):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
        
        # Time embedding processing
        # Time embedding processing
        self.Dense_0 = nn.Linear(time_dim, out_channels)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
        nn.init.zeros_(self.Dense_0.bias)
        self.act = nn.PReLU()
        
    def forward(self, x, t):
        x = self.conv(x)
        time_emb = self.Dense_0(self.act(t))[:, :, None, None]
        x = x + time_emb
        x = self.act(self.bn(x))
        return x

class TimeAwareGTConvBlock(nn.Module):
    """Group Temporal Convolution with time embedding"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, 
                 padding, dilation, use_deconv=False, time_dim=32):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                    stride=stride, padding=padding,
                                    dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2)
        
        # Time embedding processing
        self.Dense_0 = nn.Linear(time_dim, hidden_channels)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
        nn.init.zeros_(self.Dense_0.bias)
        self.act = nn.PReLU()


    def shuffle(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = rearrange(x, 'b c g t f -> b (c g) t f')
        return x

    def forward(self, x, t):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_conv1(x1)

        # Add time embedding
        time_emb = self.Dense_0(self.act(t))[:, :, None, None]
        h1 = h1 + time_emb

        h1 = self.point_act(self.point_bn1(h1))
        
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x = self.shuffle(h1, x2)
        return x


class TimeAwareDPGRNN(nn.Module):
    """Grouped Dual-path RNN with time embedding"""
    def __init__(self, input_size, width, hidden_size, time_dim, **kwargs):
        super(TimeAwareDPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        
        self.inter_rnn = GRNN(input_size=input_size*self.width, hidden_size=hidden_size*4, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size*4, self.width*hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
        
        # Time embedding processing
        self.Dense_0 = nn.Linear(time_dim, hidden_size)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
        nn.init.zeros_(self.Dense_0.bias)
        self.act = nn.PReLU()

    def forward(self, x, t):
        # Add time embedding
        time_emb = self.Dense_0(self.act(t))[:, :, None, None]
        x = x + time_emb

        ## Intra RNN
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out
        inter_x = x.reshape(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)
        return dual_out

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s

@BackboneRegister.register("dgtcrn")
class DiffusionGTCRN(nn.Module):
    def __init__(self, input_channels, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = GaussianFourierProjection(
                embedding_size=time_dim//2, scale=16
            )
        
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)

        # Modified encoder with time embedding
        self.en_convs = nn.ModuleList([
            TimeAwareConvBlock(3 * 6, 48, (1,5), stride=(1,2), padding=(0,2), 
                             use_deconv=False, is_last=False, time_dim=time_dim),
            TimeAwareConvBlock(48, 48, (1,5), stride=(1,2), padding=(0,2), 
                             groups=2, use_deconv=False, is_last=False, time_dim=time_dim),
            TimeAwareGTConvBlock(48, 48, (3,3), stride=(1,1), padding=(0,1), 
                               dilation=(1,1), use_deconv=False, time_dim=time_dim),
            TimeAwareGTConvBlock(48, 48, (3,3), stride=(1,1), padding=(0,1), 
                               dilation=(2,1), use_deconv=False, time_dim=time_dim),
            TimeAwareGTConvBlock(48, 48, (3,3), stride=(1,1), padding=(0,1), 
                               dilation=(5,1), use_deconv=False, time_dim=time_dim)
        ])
        
        self.dpgrnn1 = TimeAwareDPGRNN(48, 33, 48, time_dim=time_dim)
        self.dpgrnn2 = TimeAwareDPGRNN(48, 33, 48, time_dim=time_dim)

        # Modified decoder with time embedding
        self.de_convs = nn.ModuleList([
            TimeAwareGTConvBlock(48, 48, (3,3), stride=(1,1), padding=(2 * 5,1), 
                               dilation=(5,1), use_deconv=True, time_dim=time_dim),
            TimeAwareGTConvBlock(48, 48, (3,3), stride=(1,1), padding=(2 * 2,1), 
                               dilation=(2,1), use_deconv=True, time_dim=time_dim),
            TimeAwareGTConvBlock(48, 48, (3,3), stride=(1,1), padding=(2 * 1,1), 
                               dilation=(1,1), use_deconv=True, time_dim=time_dim),
            TimeAwareConvBlock(48, 48, (1,5), stride=(1,2), padding=(0,2), 
                             groups=2, use_deconv=True, is_last=False, time_dim=time_dim),
            TimeAwareConvBlock(48, 2, (1,5), stride=(1,2), padding=(0,2), 
                             use_deconv=True, is_last=True, time_dim=time_dim)
        ])

        self.mask = Mask()

    def forward(self, spec, time):
        """
        spec: (B, F, T, 2)
        time: (B,) time steps
        """
        # Get time embeddings
        t = self.time_embed(time)
        
        # spec_ref = spec
        # spec_real = spec[..., 0].permute(0,2,1)
        # spec_imag = spec[..., 1].permute(0,2,1)
        # spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        # feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)
        mix_spec = torch.stack([spec[:,0,:,:].real, spec[:,0,:,:].imag], -1)
        spec = torch.stack([spec[:,1,:,:].real, spec[:,1,:,:].imag], -1)

        spec_ref = mix_spec  # (B,F,T,2)

        mix_spec_real = mix_spec[..., 0].permute(0,2,1)
        mix_spec_imag = mix_spec[..., 1].permute(0,2,1)
        mix_spec_mag = torch.sqrt(mix_spec_real**2 + mix_spec_imag**2 + 1e-12)

        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
    
        feat = torch.stack([spec_mag, spec_real, spec_imag, mix_spec_mag, mix_spec_real, mix_spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb.bm(feat)
        feat = self.sfe(feat)

        # Encoder with time embedding
        en_outs = []
        x = feat
        for conv in self.en_convs:
            x = conv(x, t)
            en_outs.append(x)

        # DPGRNN blocks with time embedding
        x = self.dpgrnn1(x, t)
        x = self.dpgrnn2(x, t)

        # Decoder with time embedding
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i], t)
        
        m = self.erb.bs(x)
        spec_enh = self.mask(m, spec_ref.permute(0,3,2,1))
        spec_enh = spec_enh.permute(0,3,2,1).unsqueeze(1)  # (B,F,T,2)
        
        spec_enh = spec_enh.contiguous().to(torch.float32)
        return torch.view_as_complex(spec_enh)

if __name__ == "__main__":
    # Test configuration
    batch_size = 2
    freq_bins = 256  # Typical STFT frequency bins
    time_steps = 100  # Arbitrary number of time frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random input spectrogram (batch, freq, time, 2 for real/imag)
    input_spec = torch.randn(batch_size, freq_bins, time_steps, 2).to(device)
    
    # Random time steps (batch_size,)
    time_steps = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Initialize model
    model = DiffusionGTCRN(time_dim=32).to(device)
    
    # Forward pass
    output_spec = model(input_spec, time_steps)
    
    # Print shapes
    print(f"Input shape: {input_spec.shape}")
    print(f"Output shape: {output_spec.shape}")
    
    # Verify output shape matches input
    assert output_spec.shape == input_spec.shape, "Output shape should match input shape"
    
    # Print model summary
    print("\nModel Summary:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")