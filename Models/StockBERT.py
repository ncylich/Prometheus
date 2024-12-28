import sys
if 'google.colab' in sys.modules:
    from Prometheus.Train.train_bert import train_model
    from Prometheus.Train.dataloaders import get_long_term_Xmin_data_loaders
    from Prometheus.Models.load_config import dynamic_load_config, update_config_with_factor
else:
    from Train.train_bert import train_model
    from Train.dataloaders import get_long_term_Xmin_data_loaders
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
    nhid: int = 128
    nhead: int = 8
    dim_feedfwd: int = 512
    nlayers: int = 24
    dropout: float = 0.1
    batch_size: int = 1024
    test_col: str = 'close'

    lr = 1e-3
    epochs = 100
    init_weight_magnitude = 1e-3  # / (factor ** 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockBert(nn.Module):
    def __init__(self,
                 n_tickers,  # Number of tickers
                 embed_dim=128,  # Embedding dimension
                 n_heads=4,  # Number of attention heads
                 num_layers=2,  # Number of transformer layers
                 ff_dim=256,  # Feed-forward dimension in transformer
                 backcast_size=10,  # Dimension of continuous features (e.g. velocity vector)
                 max_seq_len=8,  # Maximum sequence length
                 n_years=20  # Number of years in the dataset
                 ):
        super().__init__()

        # Token Embeddings (for ticker IDs)
        self.token_embed = nn.Embedding(n_tickers + 1, embed_dim) # adding 1 for mask

        # Positional Embeddings
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Time Embeddings
        self.hour_embed = nn.Embedding(24, embed_dim)
        self.month_embed = nn.Embedding(12, embed_dim)
        self.year_embed = nn.Embedding(n_years, embed_dim)  # Assuming years from 2004 to 2024

        # Continuous Feature Embeddings
        # This maps continuous features to the same space as token embeddings.
        self.cont_feat_proj = nn.Linear(backcast_size, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=n_heads,
                                                   dim_feedforward=ff_dim,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLM Head: Project from hidden state back to backcast size
        self.mlm_head = nn.Linear(embed_dim, backcast_size)

    def forward(self, token_ids, cont_feats, time_indices):
        """
        token_ids: (batch_size, seq_len) LongTensor with ticker or [MASK] IDs
        cont_feats: (batch_size, seq_len, continuous_feat_dim) continuous features for each token
        time_indices: (batch_size, 3) start time of the sequence, using hour, month, and year
        """
        batch_size, seq_len = token_ids.size()

        # Token embedding
        token_embeddings = self.token_embed(token_ids)  # (B, L, D)

        # Positional embedding
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # (1, L)
        pos_embeddings = self.pos_embed(positions)  # (1, L, D)

        # Time embeddings
        hour_embeddings = self.hour_embed(time_indices[:, 0])  # (B, D)
        month_embeddings = self.month_embed(time_indices[:, 1] - 1)  # (B, D)
        year_embeddings = self.year_embed(2024 - time_indices[:, 2])  # (B, D)

        # Continuous feature embeddings
        cont_embeddings = self.cont_feat_proj(cont_feats)  # (B, L, D)
        # cont_embeddings = self.cont_feat_proj(self.normalize_velocities(cont_feats))  # (B, L, D)

        # # print all shapes
        # print('token_embeddings:', token_embeddings.shape)
        # print('cont_embeddings:', cont_embeddings.shape)
        # print('pos_embeddings:', pos_embeddings.shape)
        # print('hour_embeddings:', hour_embeddings.shape)

        # Combine embeddings: sum token embeddings, continuous embeddings, and positional embeddings
        x = token_embeddings + cont_embeddings + pos_embeddings + 0.3 * (hour_embeddings +
                                                                         month_embeddings +
                                                                         year_embeddings).unsqueeze(1)

        # Pass through transformer
        # Note: For nn.TransformerEncoder we can pass src_key_padding_mask (B,L) with True=pad
        x = self.transformer_encoder(x)

        # MLM Head
        logits = self.mlm_head(x)  # (B, L, continuous_feat_dim)

        return logits

    # Unused - integrated mean and std normalization into dataset instead
    @staticmethod
    def normalize_velocities(cont_feats):
        '''
        ['CL', 'ZN', 'SI', 'DX', 'VX', 'NG', 'HG', 'GC']
        Mean:
        tensor([1.0003, 1.0003, 1.0001, 1.0002, 1.0004, 1.0002, 1.0001, 1.0002])
        Std:
        tensor([0.0616, 0.0554, 0.0208, 0.0201, 0.0306, 0.0264, 0.0172, 0.0199])
        '''
        # cont feats is of size (batch_size, 8, continuous_feat_dim)
        mean = torch.tensor([1.0003, 1.0003, 1.0001, 1.0002, 1.0004, 1.0002, 1.0001, 1.0002]).to(device) # size (8)
        std = torch.tensor([0.0616, 0.0554, 0.0208, 0.0201, 0.0306, 0.0264, 0.0172, 0.0199]).to(device) # size (8)

        mean = mean.unsqueeze(0).unsqueeze(-1)  # size (1, 8, 1)
        std = std.unsqueeze(0).unsqueeze(-1)    # size (1, 8, 1)

        return (cont_feats - mean) / std

def main(config_path: str = ''):
    config = dynamic_load_config(config_path, Config)
    config = update_config_with_factor(config)

    train_loader, test_loader = get_long_term_Xmin_data_loaders(config.backcast_size, config.forecast_size,
                                                                config.group_len, x_min=30, batch_size=config.batch_size)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            # init.xavier_uniform_(m.weight)  # Xavier  # +/- sqrt(features)
            # init.zeros_(m.weight)  # Zeros
            init.normal_(m.weight, mean=0, std=config.init_weight_magnitude)
            if m.bias is not None:
                init.zeros_(m.bias)
                # init.normal_(m.bias, mean=0, std=init_weight_magnitude)
        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, mean=0, std=config.init_weight_magnitude)

    n_tickers = 8 if not config.group_len else 8 * config.group_len
    model = StockBert(n_tickers=n_tickers,
                      embed_dim=config.nhid,
                      n_heads=config.nhead,
                      num_layers=config.nlayers,
                      backcast_size=config.backcast_size).to(device)

    if config.init_weight_magnitude:
        model.apply(init_weights)

    optimizer = AdamW(model.parameters(), lr=config.lr)

    patience = max(1, math.floor(math.log(config.epochs, math.e)))  # floor of ln(epochs) and at least 1
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience)

    criterion = MSELoss()

    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config.epochs, device)


if __name__ == '__main__':
    main('configs/stock_bert_config.yaml')