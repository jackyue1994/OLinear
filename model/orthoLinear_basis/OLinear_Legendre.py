import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, EncoderLayer, LinearEncoder, LinearEncoder_Multihead
from layers.SelfAttention_Family import AttentionLayer, EnhancedAttention

from numpy.polynomial import Legendre as L


class LegendreTransform(torch.nn.Module):
    def __init__(self, deg=5, T=96, device='cuda:0'):
        super(LegendreTransform, self).__init__()
        self.deg = deg
        self.T = T
        self.device = device

        tvals = np.linspace(-1, 1, T)  # The Legendre series are defined in t\in[-1, 1]
        legendre_polys = np.array(
            [L.basis(i)(tvals) for i in range(self.deg)])  # Generate the basis functions which are sampled at tvals.
        # tvals = torch.from_numpy(tvals).to(device)
        self.legendre_polys = torch.from_numpy(legendre_polys).float().to(device)  # shape: [degree, T]

    def forward_transform(self, vector):
        """
        Compute the Legendre polynomial coefficients from the input vector.
        Supports input of shape [B, N, D, T], and operates on the last dimension (T).
        """
        *batch_dims, T = vector.shape
        assert T == self.T
        data = vector.reshape(-1, T).transpose(-1, -2)

        coeffs_candidate = torch.mm(self.legendre_polys, data) / T * 2
        coeffs = torch.stack([coeffs_candidate[i] * (2 * i + 1) / 2 for i in range(self.deg)]).to(self.device)
        coeffs = coeffs.transpose(-1, -2)  # shape: [-1, degree]

        return coeffs.reshape(*batch_dims, self.deg)

    def inverse_transform(self, coeffs):
        """
        Reconstruct the vector from the Legendre polynomial coefficients.
        Supports input of shape [B, N, D, self.N], and operates on the last dimension (N).
        """
        *batch_dims, N = coeffs.shape
        reshaped_coeffs = coeffs.reshape(-1, N)

        # Use precomputed polynomials
        vector = torch.matmul(reshaped_coeffs, self.legendre_polys)

        return vector.reshape(*batch_dims, -1)


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

        # chebyshev transform
        self.Legendre_degree = self.seq_len
        self.transform = LegendreTransform(self.Legendre_degree, self.seq_len)

        # output linear
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

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
            nn.Linear(self.Legendre_degree * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.Legendre_degree * self.embed_size)
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
        x_trans = self.transform.forward_transform(x)
        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.Legendre_degree)
        x = self.transform.inverse_transform(x_trans)

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
