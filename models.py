import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from mel_processing import spectrogram_torch_data
from sifigan.generator import SiFiGANGenerator

class TextEncoder(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      requires_grad=True):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)
    #パラメータを学習しない
    if requires_grad == False:
      for param in self.parameters():
        param.requires_grad = False

  def forward(self, x, x_lengths):
    x = torch.transpose(x.half(), 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask

class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0,
      requires_grad=True):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

    #パラメータを学習しない
    if requires_grad == False:
      for param in self.parameters():
        param.requires_grad = False

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0,
      requires_grad=True):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    #パラメータを学習しない
    if requires_grad == False:
      for param in self.parameters():
        param.requires_grad = False


  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, 
          initial_channel, 
          resblock, 
          resblock_kernel_sizes, 
          resblock_dilation_sizes, 
          upsample_rates, 
          upsample_initial_channel, 
          upsample_kernel_sizes, 
          requires_grad=True):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if requires_grad == False:
          for param in self.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        #periods = [2,3,5,7,11]
        periods = [3,5,7,11,13]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat, flag = True):
        if flag:
            y_d_rs = []
            y_d_gs = []
            fmap_rs = []
            fmap_gs = []
            for i, d in enumerate(self.discriminators):
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
                y_d_rs.append(y_d_r)
                y_d_gs.append(y_d_g)
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

            return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        else:
            y_d_gs = []
            with torch.no_grad():
                for i, d in enumerate(self.discriminators):
                    y_d_g, _ = d(y_hat)
                    y_d_gs.append(y_d_g)

            return y_d_gs            

class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)
            
        return mel_slices
    
    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:,-partial_frames:]
        
        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:,s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
        
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            #embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        
        return embed


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_flow,
    gin_channels,
    ssl_dim,
    requires_grad_pe=True,
    requires_grad_flow=True,
    requires_grad_text_enc=True,
    requires_grad_dec=True,
    requires_grad_emb_g=True,):

    super().__init__()
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.gin_channels = gin_channels
    self.requires_grad_pe = requires_grad_pe
    self.requires_grad_flow = requires_grad_flow
    self.requires_grad_text_enc = requires_grad_text_enc
    self.requires_grad_dec = requires_grad_dec
    self.requires_grad_emb_g = requires_grad_emb_g

    self.enc_p = PosteriorEncoder(
        ssl_dim,
        inter_channels, 
        hidden_channels, 
        5, 
        1, 
        16,  
        requires_grad=requires_grad_pe
        )
    self.enc_q = PosteriorEncoder(
        spec_channels, 
        inter_channels, 
        hidden_channels, 
        5, 
        1, 
        16, 
        gin_channels=gin_channels,
        requires_grad=requires_grad_text_enc
        )
    self.dec = Generator(
        inter_channels, 
        resblock, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes,
        requires_grad=requires_grad_dec
        )
    self.flow = ResidualCouplingBlock(
        inter_channels, 
        hidden_channels, 
        5, 
        1, 
        4, 
        n_flows=n_flow, 
        gin_channels=gin_channels,
        requires_grad=requires_grad_flow
        )
    self.enc_spk = SpeakerEncoder(
        model_hidden_size=gin_channels, 
        model_embedding_size=gin_channels
        )

  def forward(self, c, spec, mel=None, c_lengths=None, spec_lengths=None):
    #spk enc
    g = self.enc_spk(mel.transpose(1,2))
    g = g.unsqueeze(-1)

    #text enc
    _, m_p, logs_p, _ = self.enc_p(c, c_lengths)

    #PE
    z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g) 

    #Flow
    z_p = self.flow(z, spec_mask, g=g)

    #dec
    z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)

    return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, c, g=None, mel=None, c_lengths=None):
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    if not self.use_spk:
      g = self.enc_spk.embed_utterance(mel.transpose(1,2))
    g = g.unsqueeze(-1)

    z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
    z = self.flow(z_p, c_mask, g=g, reverse=True)
    o = self.dec(z * c_mask, g=g)
    
    return o

  def make_random_target_sids(self, target_ids, sid):
    # target_sids は target_ids をランダムで埋める
    target_sids = torch.zeros_like(sid)
    for i in range(len(target_sids)):
      source_id = sid[i]
      deleted_target_ids = target_ids[target_ids != source_id] # source_id と target_id が同じにならないよう sid と同じものを削除
      if len(deleted_target_ids) >= 1:
        target_sids[i] = deleted_target_ids[torch.randint(len(deleted_target_ids), (1,))]
      else:
        # target_id 候補が無いときは仕方ないので sid を使う
        target_sids[i] = source_id
    return target_sids

  def voice_conversion(self, y, y_lengths, sin, d, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, _, _, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(sin, z_hat * y_mask, d, sid=g_tgt)
    return o_hat

  def voice_ra_pa_db(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    o_hat = self.dec(z * y_mask, g=g_tgt)
    return o_hat, y_mask, (z)

  def voice_ra_pa_da(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    o_hat = self.dec(z * y_mask, g=g_src)
    return o_hat, y_mask, (z)

  def voice_conversion_cycle(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    z_p_hat = self.flow(z_hat, y_mask, g=g_tgt)
    z_hat_hat = self.flow(z_p_hat, y_mask, g=g_src, reverse=True)
    o_hat = self.dec(z_hat_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

  def save_synthesizer(self, path):
    enc_q = self.enc_q.state_dict()
    dec = self.dec.state_dict()
    emb_g = self.emb_g.state_dict()
    torch.save({'enc_q': enc_q,'dec': dec, 'emb_g': emb_g}, path)

  def load_synthesizer(self, path):
    dict = torch.load(path, map_location='cpu')
    enc_q = dict['enc_q']
    dec = dict['dec']
    emb_g = dict['emb_g']
    self.enc_q.load_state_dict(enc_q)
    self.dec.load_state_dict(dec)
    self.emb_g.load_state_dict(emb_g)

