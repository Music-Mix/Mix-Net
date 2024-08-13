import torch
from torch import nn
import torch.nn.functional as F

from utils.spec import stft, istft
from utils.noise import gaussian_noise, markov_noise, random_walk_noise, spectral_folding


def positional_encoding(batch_size, n_time, n_feature, zero_pad=False, scale=False, dtype=torch.float32):
    pos_indices = torch.tile(torch.unsqueeze(torch.arange(n_time), 0), [batch_size, 1])

    pos = torch.arange(n_time, dtype=dtype).reshape(-1, 1)
    pos_enc = pos / torch.pow(10000, 2 * torch.arange(0, n_feature, dtype=dtype) / n_feature)
    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])

    if zero_pad:
        pos_enc = torch.cat([torch.zeros(size=[1, n_feature]), pos_enc[1:, :]], 0)

    outputs = F.embedding(pos_indices, pos_enc)

    if scale:
        outputs = outputs * (n_feature ** 0.5)

    return outputs


class FeedForward(nn.Module):
    def __init__(self, n_feature=2048, n_hidden=512, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(n_feature, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_feature)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_layer = nn.LayerNorm(n_feature)

    def forward(self, x):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        y = self.norm_layer(y)
        return y


class TimbreBlock(nn.Module):
    def __init__(self, n_feat, nhead=8, n_layers=5, dropout=0.5, pr=0.02):
        super().__init__()
        self.n_feat = n_feat
        self.n_layers = n_layers
        self.pr = pr

        self.attn_layer1 = nn.ModuleList()
        self.attn_layer2 = nn.ModuleList()
        self.ff_layer1 = nn.ModuleList()
        self.ff_layer2 = nn.ModuleList()
        for _ in range(self.n_layers):
            _attn_layer1 = nn.MultiheadAttention(n_feat, nhead, batch_first=True)
            _attn_layer2 = nn.MultiheadAttention(n_feat, nhead, batch_first=True)
            _ff_layer1 = FeedForward(n_feat, dropout=dropout)
            _ff_layer2 = FeedForward(n_feat, dropout=dropout)
            self.attn_layer1.append(_attn_layer1)
            self.attn_layer2.append(_attn_layer2)
            self.ff_layer1.append(_ff_layer1)
            self.ff_layer2.append(_ff_layer2)

        self.norm_layer = nn.LayerNorm(n_feat)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_feat, n_feat)

    def forward(self, x, x1):
        # x: batch, n_time, n_feat
        B, T, Ft = x.shape
        x += positional_encoding(B, T, Ft) * self.pr
        B1, T1, Ft1 = x1.shape
        x1 += positional_encoding(B1, T1, Ft1) * self.pr

        for _attn_layer1, _attn_layer2, _ff_layer1, _ff_layer2 in zip(self.attn_layer1,
                                                                      self.attn_layer2,
                                                                      self.ff_layer1,
                                                                      self.ff_layer2):
            x, _ = _attn_layer1(x, x, x, need_weights=False)
            x1, _ = _attn_layer2(x, x1, x1, need_weights=False)
            x = _ff_layer1(x)
            x1 = _ff_layer2(x1)

        x = torch.stack([x, x1], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.norm_layer(x)
        x1 = x[:, 1, ]
        x = x[:, 0, ]
        return x, x1


class Encoder(nn.Module):
    def __init__(self,
                 n_feat,
                 n_stft,
                 out_feat,
                 n_block=5,
                 nhead=8,
                 n_layers=5,
                 dropout=0.25,
                 pr=0.02):
        super().__init__()

        self.n_block = n_block

        self.conv_pre1 = nn.Sequential(
            nn.ConvTranspose1d(n_stft, n_feat, kernel_size=1),
            nn.BatchNorm1d(n_feat),
            nn.LeakyReLU(0.01),
        )
        self.conv_pre2 = nn.Sequential(
            nn.ConvTranspose1d(n_feat, n_feat, kernel_size=1),
            nn.BatchNorm1d(n_feat),
            nn.LeakyReLU(0.01),
        )
        self.layers = nn.ModuleList()
        for i in range(n_block):
            self.layers.append(TimbreBlock(n_feat, nhead=nhead, n_layers=n_layers, dropout=dropout, pr=pr))

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat * 2, n_feat, kernel_size=1),
            nn.BatchNorm1d(n_feat),
            nn.ReLU(),
            nn.Conv1d(n_feat, n_feat, kernel_size=1),
            nn.BatchNorm1d(n_feat),
            nn.ReLU(),
        )
        self.linear = nn.Linear(n_feat, out_feat)

    def forward(self, x, x1):
        x = self.conv_pre1(x)
        x1 = x1.transpose(-1, -2)
        x1 = self.conv_pre2(x1)
        x = x.transpose(-1, -2)
        x1 = x1.transpose(-1, -2)
        for layer in self.layers:
            x, x1 = layer(x, x1)

        z = torch.dstack([x, x1])
        z = z.transpose(-2, -1)
        z = self.conv(z)
        z = z.transpose(-2, -1)
        z = self.dropout(z)
        logits = self.linear(z)
        return logits, z


class DecoderBlock(nn.Module):

    def __init__(self, d_model, out_feat, nhead, n_layers, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers

        self.attn1_layer = nn.ModuleList()
        self.attn2_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        for _ in range(n_layers):
            _layer1 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            _layer2 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            _layer3 = FeedForward(d_model, dropout=dropout)
            self.attn1_layer.append(_layer1)
            self.attn2_layer.append(_layer2)
            self.ff_layer.append(_layer3)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.01),
            nn.Linear(d_model, out_feat)
        )

    def forward(self, x, x1):
        for i in range(self.n_layers):
            _attn1_layer = self.attn1_layer[i]
            _attn2_layer = self.attn2_layer[i]
            _ff_layer = self.ff_layer[i]
            x, _ = _attn1_layer(x, x, x, need_weights=False)
            x, _ = _attn2_layer(x, x1, x1, need_weights=False)
            x = _ff_layer(x)

        output = self.dropout(x)
        output = self.fc(output)
        return output


class Decoder(nn.Module):

    def __init__(self,
                 n_feat,
                 n_timbre,
                 n_pitch,
                 nfft=1024,
                 d_model=512,
                 nhead=8,
                 n_layers=5,
                 dropout=0.5,
                 pr=0.01):
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.pr = pr

        self.emb_feat = nn.Linear(n_feat, d_model)
        self.emb_timbre = nn.Linear(n_timbre, d_model)
        self.emb_pitch = nn.Linear(n_pitch, d_model)

        self.lrelu = nn.LeakyReLU(0.01, inplace=True)

        n_out = nfft // 2 + 1
        self.dec = DecoderBlock(d_model, n_out, nhead, n_layers, dropout=dropout)

        self.exp_layer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            nn.LeakyReLU(0.01),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(0.01),
            nn.Conv1d(d_model, n_out, kernel_size=1, bias=True),
            nn.BatchNorm1d(n_out),
            nn.LeakyReLU(0.01),
        )
        self.mask_layer = nn.Sequential(
            nn.Linear(n_out, n_out),
            nn.LeakyReLU(0.01),
            nn.Linear(n_out, n_out),
            nn.LeakyReLU(0.01),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, pitch):
        x = self.emb_timbre(x1) + self.emb_feat(x2)
        x = x * self.d_model ** (1 / 2)
        x = self.lrelu(x)
        y = x.transpose(-2, -1)
        y = self.exp_layer(y)

        x2 = self.emb_pitch(pitch)
        x += positional_encoding(x.shape[0], x.shape[1], x.shape[2]) * self.pr + self.pr
        x = self.dec(x, x2)

        y = y.transpose(-2, -1)
        z = y * x

        m = self.mask_layer(z)
        m = m.transpose(-2, -1)
        return m, z, x


class MixNet(nn.Module):

    def __init__(self,
                 n_hidden,
                 n_timbre,
                 nfft=1024,
                 hop_length=160,
                 samplerate=16000,
                 n_feat=768,
                 n_pitch=360,
                 n_tb_head=8,
                 n_tb_layers=5,
                 tb_dropout=0.5,
                 pr=0.02,
                 noise_type=None,
                 noise_kw=None,
                 ):
        super().__init__()

        self.hop_length = hop_length
        self.nfft = nfft
        self.samplerate = samplerate
        self.noise_type = noise_type
        self.noise_kw = noise_kw

        self.encoder = Encoder(
            n_feat,
            (nfft // 2) + 1,
            n_timbre,
            nhead=n_tb_head,
            n_layers=n_tb_layers,
            dropout=tb_dropout,
            pr=pr)
        self.dec_p = Decoder(
            n_feat=n_feat,
            n_timbre=n_timbre,
            n_pitch=n_pitch,
            d_model=n_hidden,
            nhead=8,
            n_layers=5,
            dropout=0.5,
        )

    def _stft(self, x):
        hl = self.hop_length
        nfft = self.nfft
        return stft(x, fft_size=nfft, hop_length=hl)

    def _istft(self, x, n_time):
        hl = self.hop_length
        return istft(x, hop_length=hl, signal_length=n_time)

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _noise(self, x):
        if self.noise_type is None:
            x = x
        else:
            noise_kw = self.noise_kw or {}
            if self.noise_type == 'gaussian':
                x = gaussian_noise(x, **noise_kw)
            elif self.noise_type == 'markov':
                x = markov_noise(x, **noise_kw)
            elif self.noise_type == 'random_walk':
                x = random_walk_noise(x, **noise_kw)
            elif self.noise_type == 'spectral_folding':
                x = spectral_folding(x, **noise_kw)
        return x

    def forward(self, audio, feat, f0, is_noise=False):
        n_len = audio.shape[-1]
        x = self._stft(audio)
        _mag = x.abs()
        if is_noise:
            _mag = self._noise(_mag)

        _, f0_t, f0_f = f0.shape
        _, ft_t, ft_f = feat.shape
        if f0_t < ft_t:
            feat = feat[:, :f0_t, :]
        elif f0_t > ft_t:
            feat = F.pad(feat, (0, 0, 0, f0_t - ft_t))

        logits, h = self.encoder(_mag, feat)
        m, z, y = self.dec_p(logits, feat, f0)

        z = x * m
        o = self._istft(z, n_len)
        o = o[..., :n_len]
        return o, z, (y, m, logits, h)
