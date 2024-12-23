import torch
import torch.nn as nn
import torch.nn.functional as F
from ns_layers.Transformer_EncDec import Encoder, EncoderLayer
from ns_layers.SelfAttention_Family import FullAttention, AttentionLayer
from ns_layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # 96
        self.pred_len = configs.pred_len    # 96
        self.output_attention = configs.output_attention    # False
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy    # 'projection'
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,    # 512
                    configs.d_ff,   # 512
                    dropout=configs.dropout,    # 0.1
                    activation=configs.activation   # gelu
                ) for l in range(configs.e_layers)  # e_layers=3
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)  # d_model=512
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):   # # x_enc:(16, 96, 862), x_mark_enc:(16, 96, 4), x_dec:(16, 114, 862), x_mark_dec:(16, 144, 4)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()    # (16, 1, 862) 
            x_enc = x_enc - means   # (16, 96, 862)
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)    # (16, 1, 862)
            x_enc /= stdev  # (16, 96, 862)

        _, _, N = x_enc.shape # B L N       # N： 862
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (16, 866, 512) covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # (16, 866, 512)  [None, None, None, None]

        # B N E -> B N S -> B S N
        # here decoder is not used, only used a projector 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates   # (16, 866, 512) ->(16, 866, 96) -> (16, 96, 866) -> (16, 96, 862)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))   # dec_out:(16, 96, 862),  stdev: (16, 862) -> (16, 1, 862) -> (16, 96, 862)
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))   # (16, 96, 862)

        return dec_out  # (16, 96, 862)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None): # x_enc: batch_x, x_mark_enc: batch_x_mark, x_dec: dec_inp, x_mark_dec: batch_y_mark
        # itransformer only have transformer encoder, no decoder, so x_dec and x_mark_dec are not used.
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)   # x_enc:(16, 96, 862), x_mark_enc:(16, 96, 4), x_dec:(16, 144, 862), x_mark_dec:(16, 144, 4)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D] (16, 96, 862)