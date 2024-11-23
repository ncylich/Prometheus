import sys
if 'google.colab' in sys.modules:
    from Prometheus.Train.train_encoder_decoder import train_model, get_long_term_data_loaders
    from Prometheus.Models.load_config import dynamic_load_config, update_config_with_factor
else:
    from Train.train_encoder_decoder import train_model, get_long_term_data_loaders
    from Models.load_config import dynamic_load_config, update_config_with_factor

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


class TriplePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, feature_types: int, n_tickers: int, max_time_steps: int = 24, dropout: float = 0.1, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.feature_types = feature_types

        # Embeddings for feature types (e.g., price and volume)
        self.feature_type_encoding = nn.Embedding(feature_types, d_model // 3).to(device)
        # Embeddings for time steps (e.g., hours of the day)
        self.time_encoding = nn.Embedding(max_time_steps, d_model // 3).to(device)
        # Embeddings for tickers
        self.ticker_encoding = nn.Embedding(n_tickers, d_model // 3).to(device)

    def forward(self, x: torch.Tensor, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
            time_indices: Tensor of shape [batch_size], indices of time steps
        """
        num_features, batch_size, d_model = x.size()
        n_tickers = num_features // self.feature_types
        third = x.size(2) // 3

        # Add feature type encoding
        x[:, :, 0:third*3:3] = x[:, :, 0:third*3:3] + self.feature_type_encoding(torch.arange(self.feature_types, device=self.device)).repeat_interleave(n_tickers, axis=0).unsqueeze(1)

        # Add time encoding
        x[:, :, 1:third*3:3] = x[:, :, 1:third*3:3] + self.time_encoding(time_indices.to(self.device)).unsqueeze(0)

        # Add ticker encoding
        x[:, :, 2:third*3:3] = x[:, :, 2:third*3:3] + self.ticker_encoding(torch.arange(n_tickers, device=self.device)).repeat(self.feature_types, 1).unsqueeze(1)

        return self.dropout(x)

class SinusoidalPositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(SinusoidalPositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, d_model = x.size()
        # [batch_size = 128, seq_len = 30]

        return x + self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class StaticPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device='cuda:0'):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.pe(pos)

class EncoderDecoder(nn.Module):
    def __init__(self, in_F, out_F, nhid=256, nhead=8, dim_feedfwd=1024, enc_layers=6, dec_layers=6, group_size=4,
                 dropout=0.1, activation='gelu', init_weight_magnitude = 1e-2, device='cuda:0', feature_types=2,
                 n_tickers=8, max_time_steps=24):
        super(EncoderDecoder, self).__init__()

        assert in_F % group_size == 0, f"group size, {group_size}, must divide forecast size, {in_F}"

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
        # self.pos_decoder = SinusoidalPositionalEncoding(d_model=nhid, max_len=out_F // group_size, device=device)
        self.pos_decoder = StaticPositionalEmbedding(d_model=nhid, max_len=out_F // group_size, device=device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_feedfwd, dropout=dropout, activation=activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_feedfwd, dropout=dropout, activation=activation)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.init_weight_magnitude = init_weight_magnitude
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -self.init_weight_magnitude, self.init_weight_magnitude)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -self.init_weight_magnitude, self.init_weight_magnitude)
        elif isinstance(m, nn.TransformerEncoderLayer) or isinstance(m, nn.TransformerDecoderLayer):
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # TODO: fixed decoder to produce 1-token at a time
    def forward(self, src, time_indices, tgt=None):
        """
        Args:
            src: Tensor of shape [B, V, in_F]
            tgt: Tensor of shape [B, out_F, X]
            time_indices: Tensor of shape [B, seq_len]
        """
        if tgt is None:
            tgt = torch.zeros(src.size(0), self.out_F // self.group_size, self.group_size).to(self.device)

        # Encoder
        src = src.transpose(0, 1)  # [V, B, in_F]
        src = self.encoder_input_projection(src)  # [V, B, nhid]
        src = self.pos_encoder(src, time_indices)
        memory = self.encoder(src)  # [V, B, nhid]

        # Decoder
        tgt = self.decoder_input_projection(tgt)  # [out_F, B, nhid]
        tgt = self.pos_decoder(tgt).transpose(0, 1)  # [out_F, B, nhid]
        output = self.decoder(tgt, memory)  # [out_F, B, nhid]

        # Output projection
        output = self.output_projection(output)  # [out_F, B, X]
        output = output.permute(1, 0, 2)  # [B, out_F, X]
        B, out_F, X = output.size()
        output = output.reshape(B, out_F * X)

        return output

def main(config_path: str = ''):
    config = dynamic_load_config(config_path, Config)
    config = update_config_with_factor(config)

    # data_loader, test_loader = get_old_data_loaders(config.backcast_size, config.forecast_size, test_size_ratio=0.2,
    #                                             batch_size=config.batch_size, dataset_col=config.test_col)

    data_loader, test_loader = get_long_term_data_loaders(config.backcast_size, config.forecast_size, test_size_ratio=0.2,
                                          batch_size=config.batch_size, dataset_col=config.test_col, truncate_data=True)

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

if __name__ == '__main__':
    main('configs/encoder_decoder_config.yaml')