import torch
import torch.nn as nn
from ns_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from ns_layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats): # x (32, 96, 321)  stats (32, 1, 321)
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E  32, 1, 321)
        x = torch.cat([x, stats], dim=1)  # B x 2 x E    (32, 2, 321)
        x = x.view(batch_size, -1)  # B x 2E  (32, 642)
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

        self.z_mean = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )
        self.z_logvar = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )

        self.z_out = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                               torch.exp(posterior_logvar), dim=1)
        return torch.mean(KL)

    def reparameterize(self, posterior_mean, posterior_logvar):
        posterior_var = posterior_logvar.exp()
        # take sample
        if self.training:
            posterior_mean = posterior_mean.repeat(100, 1, 1, 1)
            posterior_var = posterior_var.repeat(100, 1, 1, 1)
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            z = z.mean(0)
        else:
            z = posterior_mean
        # z = posterior_mean
        return z

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc -> batch_x, x_mark_enc: batch_x_mark, x_dec: dec_inp, x_mark_dec: batch_y_mark
        # x_enc (32, 96, 7), x_mark_enc (32, 96, 4), x_dec (32, 240, 7), x_mark_dec (32, 240, 4)
        x_raw = x_enc.clone().detach()  # (32, 96, 7)

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E , (32, 1, 7)
        x_enc = x_enc - mean_enc    # (32, 96, 7)
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E (32, 1, 7)
        x_enc = x_enc / std_enc # (32, 96, 7)
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()   # (32, 240, 7)

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar (32, 1)
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S (32, 96)

        # Model Inference -> reparameterize
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (32, 96, 512)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)   # enc_out: # (32, 96, 512), attns: [None, None]

        mean = self.z_mean(enc_out) # (32, 96, 512)
        logvar = self.z_logvar(enc_out) # (32, 96, 512)

        z_sample = self.reparameterize(mean, logvar)   # (32, 96, 512)

        # dec_out = self.z_out(torch.cat([z_sample, dec_out], dim=-1))
        enc_out = self.z_out(z_sample)  # (32, 96, 512)
        # calculate KL loss between z_sample and gaussian distribution
        KL_z = self.KL_loss_normal(mean, logvar)    # tensor(0.0391, device='cuda:6', grad_fn=<MeanBackward0>)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec) # (32, 240, 512)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta) # (32, 240, 7)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc  # (32, 240, 7)

        if self.output_attention:   # false
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], dec_out, KL_z, z_sample  # [B, L, D] (32, 192, 7), (32, 240, 7), tensor(0.0391, device='cuda:6', grad_fn=<MeanBackward0>) # (32, 96, 512)
