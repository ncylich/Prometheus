import sys
if 'google.colab' in sys.modules:
    from Prometheus.Train.train_encoder_decoder import train_model, get_original_data_loaders, get_long_term_data_loaders
    from Prometheus.Models.load_config import dynamic_load_config, update_config_with_factor
    from Prometheus.Models.Somoformer import TriplePositionalEncoding
else:
    from Train.train_encoder_decoder import train_model, get_original_data_loaders, get_long_term_data_loaders
    from Models.load_config import dynamic_load_config, update_config_with_factor
    from Models.Somoformer import TriplePositionalEncoding

from enum import Enum
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Config:
    forecast_size: int = 36
    backcast_size: int = 36 * 2

    factor: int = 2
    seq_len: int = backcast_size + forecast_size
    nhid: int = 128 * factor
    nhead: int = 8
    dim_feedfwd: int = 512 * factor
    n_enc_layers: int = 12
    n_dec_layers: int = 12
    n_values_per_group: int = 1
    dropout: float = 0.1
    batch_size: int = 1024
    test_col: str = 'close'

    lr = 1e-3
    epochs = 100
    init_weight_magnitude = 1e-3  # / (factor ** 2)

    aux_loss_weight: float = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dct_matrix(N):
    """Calculates DCT Matrix of size N."""
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, in_F, out_F, nhid=256, nhead=8, dim_feedfwd=1024, enc_layers=6, dec_layers=6, group_size=4,
                 dropout=0.1, activation='gelu', init_weight_magnitude = 1e-2, device='cuda:0', feature_types=2,
                 n_tickers=8, max_time_steps=24):
        super(EncoderDecoder, self).__init__()

        self.device = device
        self.in_F = in_F
        self.out_F = out_F
        self.group_size = group_size

        self.encoder_input_projection = nn.Linear(in_F, nhid)
        self.decoder_input_projection = nn.Linear(group_size, nhid)
        self.output_projection = nn.Linear(nhid, group_size)

        self.pos_encoder = TriplePositionalEncoding(
            d_model=nhid,
            feature_types=feature_types,
            n_tickers=n_tickers,
            max_time_steps=max_time_steps,
            dropout=dropout,
            device=device
        )
        self.pos_decoder = PositionalEncoding(max_seq_len=out_F, embed_model_dim=nhid)

        encoder_layer = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_feedfwd, dropout=dropout, activation=activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_feedfwd, dropout=dropout, activation=activation)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

    def forward(self, src, time_indices, tgt=None):
        """
        Args:
            src: Tensor of shape [B, V, in_F]
            tgt: Tensor of shape [B, out_F, X]
            time_indices: Tensor of shape [B, seq_len]
        """
        if tgt is None:
            tgt = torch.zeros(src.size(0), self.out_F, self.group_size).to(self.device)

        src = src.transpose(0, 1)  # [V, B, in_F]
        tgt = tgt.transpose(0, 1) # [out_F, B, X]
        V, B, in_F = src.size()
        out_F, B, X = tgt.size()

        # Encoder
        src = self.encoder_input_projection(src)  # [V, B, nhid]
        src = self.pos_encoder(src, time_indices)
        memory = self.encoder(src)  # [V, B, nhid]

        # Decoder
        tgt = self.decoder_input_projection(tgt)  # [out_F, B, nhid]
        tgt = self.pos_decoder(tgt)
        output = self.decoder(tgt, memory)  # [out_F, B, nhid]

        # Output projection
        output = self.output_projection(output)  # [out_F, B, X]
        output = output.permute(1, 0, 2)  # [B, out_F, X]

        return output

def main(config_path: str = ''):
    config = dynamic_load_config(config_path, Config)
    config = update_config_with_factor(config)

    # data_loader, test_loader = get_old_data_loaders(config.backcast_size, config.forecast_size, test_size_ratio=0.2,
    #                                             batch_size=config.batch_size, dataset_col=config.test_col)

    data_loader, test_loader = get_long_term_data_loaders(config.backcast_size, config.forecast_size, test_size_ratio=0.2,
                                                          batch_size=config.batch_size, dataset_col=config.test_col)

    model = EncoderDecoder(config.backcast_size,
                           config.forecast_size,
                           nhid=config.nhid,
                           nhead=config.nhead,
                           dim_feedfwd=config.dim_feedfwd,
                           enc_layers=config.n_enc_layers,
                           dec_layers=config.n_dec_layers,
                           group_size=config.n_values_per_group,
                           dropout=config.dropout,
                           init_weight_magnitude=config.init_weight_magnitude,
                           device=str(device)).to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr)

    patience = max(1, math.floor(math.log(config.epochs, math.e))) # floor of ln(epochs) and at least 1
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience)

    def loss_function(y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    train_model(model, data_loader, test_loader, loss_function, optimizer, scheduler, config.epochs)