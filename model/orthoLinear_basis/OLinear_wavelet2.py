import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, EncoderLayer, LinearEncoder, LinearEncoder_Multihead
from layers.SelfAttention_Family import AttentionLayer, EnhancedAttention

import pywt


class DifferentiableWaveletTransform(torch.nn.Module):
    def __init__(self, wavelet='dmey', T=None, device='cuda:0'):
        super(DifferentiableWaveletTransform, self).__init__()
        self.device = device
        self.T = T
        assert self.T is not None
        self.wavelet_len = T // 2

        # Load wavelet filters
        wavelet = pywt.Wavelet(wavelet)
        self.filter_size = len(wavelet.dec_lo)  # 保存滤波器长度
        self.dec_lo = torch.tensor(wavelet.dec_lo, dtype=torch.float32, device=device).view(1, 1, -1)
        self.dec_hi = torch.tensor(wavelet.dec_hi, dtype=torch.float32, device=device).view(1, 1, -1)
        self.rec_lo = torch.tensor(wavelet.rec_lo, dtype=torch.float32, device=device).view(1, 1, -1)
        self.rec_hi = torch.tensor(wavelet.rec_hi, dtype=torch.float32, device=device).view(1, 1, -1)

    def forward_transform(self, vector):
        """Compute wavelet coefficients using convolution."""
        B, N, D, T = vector.shape  # Get original dimensions
        assert self.T == T
        vector = vector.reshape(B * N, D, T)  # Reshape to [batch_size, in_channels, signal_length]

        # Expand filters to match input channels
        dec_lo = self.dec_lo.expand(D, -1, -1)  # Shape: [D, 1, filter_size]
        dec_hi = self.dec_hi.expand(D, -1, -1)  # Shape: [D, 1, filter_size]

        # Calculate padding for 'same' convolution
        padding = self.filter_size // 2

        # Perform convolution for decomposition
        cA = F.conv1d(vector, dec_lo, stride=2, padding=padding, groups=D)
        cD = F.conv1d(vector, dec_hi, stride=2, padding=padding, groups=D)

        # Restore the batch dimensions
        cA = cA.reshape(B, N, D, -1)[..., :self.wavelet_len]
        cD = cD.reshape(B, N, D, -1)[..., :self.wavelet_len]

        return cA, cD

    def inverse_transform(self, cA, cD):
        """Reconstruct the signal using transposed convolution."""
        B, N, D, T_half = cA.shape  # Get original dimensions
        cA = cA.reshape(B * N, D, T_half)
        cD = cD.reshape(B * N, D, T_half)

        # Expand filters to match input channels
        rec_lo = self.rec_lo.expand(D, -1, -1)  # Shape: [D, 1, filter_size]
        rec_hi = self.rec_hi.expand(D, -1, -1)  # Shape: [D, 1, filter_size]

        # Calculate padding for transposed convolution
        padding = self.filter_size // 2 - 1

        # Perform transposed convolution for reconstruction
        up_cA = F.conv_transpose1d(cA, rec_lo, stride=2, padding=padding, groups=D)
        up_cD = F.conv_transpose1d(cD, rec_hi, stride=2, padding=padding, groups=D)

        # Correct the reconstructed signal length
        expected_length = up_cA.shape[-1]  # Length after transposed convolution
        #         original_length = 2 * T_half  # Original signal length
        if expected_length > self.T:
            up_cA = up_cA[..., :self.T]
            up_cD = up_cD[..., :self.T]

        # Sum the reconstructed signals
        reconstructed = up_cA + up_cD

        # Restore the batch dimensions
        reconstructed = reconstructed.reshape(B, N, D, -1)

        return reconstructed[..., :self.T]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff

        # self.channel_independence = configs.channel_independence
        self.embed_size = configs.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        # output linear
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # wavelet transform
        self.wavelet_length = self.seq_len // 2
        self.transform = DifferentiableWaveletTransform(T=self.seq_len)

        # #############  transformer related  #########
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=configs.d_model, d_ff=configs.d_ff, CovMat=None,
                    dropout=configs.dropout, activation=configs.activation, token_num=self.enc_in,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag
        )
        self.ortho_trans = nn.Sequential(
            nn.Linear(self.wavelet_length * self.embed_size * 2, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.wavelet_length * self.embed_size * 2)
        )

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Trans(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # orthogonal transformation
        # [B, N, D, T]
        cA, cB = self.transform.forward_transform(vector=x)
        cAB = torch.cat([cA, cB], dim=-1)
        cAB = self.ortho_trans(cAB.flatten(-2)).reshape(B, N, D, self.wavelet_length * 2)
        x = self.transform.inverse_transform(cAB[..., :self.wavelet_length], cAB[..., self.wavelet_length:])

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, tau, D]
        x = self.Fre_Trans(x) + x

        # linear
        # [B, N, T*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
