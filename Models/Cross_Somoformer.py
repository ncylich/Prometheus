from torchvision.ops.misc import interpolate
import sys
if 'google.colab' in sys.modules:
    from Prometheus.Train.train_somoformer import train_model, get_data_loaders
    from Prometheus.Models.load_config import dynamic_load_config
else:
    from Train.train_somoformer import train_model, get_data_loaders
    from Models.load_config import dynamic_load_config

from enum import Enum
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import math
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
from dataclasses import dataclass

# TODO: fix positional encoding and token breakdown for input and output space

# Parameters

@dataclass
class Config:
    forecast_size: int = 36
    backcast_size: int = 36 * 2

    factor: int = 2
    nhid: int = 128 * factor
    nhead: int = 8
    dim_feedfwd: int = 512 * factor
    nlayers: int = 24
    dropout: float = 0.1
    batch_size: int = 1024
    test_col: str = 'close'

    lr = 1e-3
    epochs = 100
    init_weight_magnitude = 1e-3  # / (factor ** 2)

    aux_loss_weight: float = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TriplePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, feature_types: int, n_tickers: int, max_time_steps: int = 5000, dropout: float = 0.1, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.feature_types = feature_types

        # Embeddings for feature types (e.g., price and volume)
        self.feature_type_encoding = nn.Embedding(feature_types, d_model // 3).to(device)
        # Embeddings for time steps
        self.time_encoding = nn.Embedding(max_time_steps, d_model // 3).to(device)
        # Embeddings for tickers
        self.ticker_encoding = nn.Embedding(n_tickers, d_model // 3).to(device)


    def forward(self, x: torch.Tensor, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
            time_indices: Tensor of shape [seq_len], indices of time steps
        """
        seq_len, batch_size, d_model = x.size()
        n_tickers = seq_len // self.feature_types  # for case of variable number of tickers
        third = d_model // 3

        # Prepare indices
        # Add feature type encoding
        x[:, :, 0:third*3:3] = x[:, :, 0:third*3:3] + self.feature_type_encoding(torch.arange(self.feature_types, device=self.device)).repeat_interleave(n_tickers, axis=0).unsqueeze(1)

        # Add time encoding
        x[:, :, 1:third*3:3] = x[:, :, 1:third*3:3] + self.time_encoding(time_indices.to(self.device)).unsqueeze(0)

        # Add ticker encoding
        x[:, :, 2:third*3:3] = x[:, :, 2:third*3:3] + self.ticker_encoding(torch.arange(n_tickers, device=self.device)).repeat(self.feature_types, 1).unsqueeze(1)

        return self.dropout(x)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, fc_over_bc, d_model, nhead, dim_feedforward, dropout, activation, attention_type):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.forecast_dim = fc_over_bc * d_model
        self.attention_type = attention_type  # 'cross' or 'self'

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: [seq_len, batch_size, d_model]

        if self.attention_type == 'cross':
            # Cross-attention: future tokens attend to past tokens
            past_tokens = src[:src.size(0) - self.forecast_dim]
            future_tokens = src[src.size(0) - self.forecast_dim:]

            # Allow future tokens to attend to past tokens
            attn_output, _ = self.self_attn(future_tokens, past_tokens, past_tokens)
            src2 = torch.cat([torch.zeros_like(past_tokens), attn_output], dim=0)
        else:
            # Self-attention among all tokens
            src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class Somoformer(nn.Module):
    def __init__(self, backcast_size, forecast_size, nhid=256, nhead=8, dim_feedfwd=1024, nlayers=6, dropout=0.1,
                 activation='gelu', init_weight_magnitude=1e-3, device='cuda:0', feature_types=2, n_tickers=7, max_time_steps=5000):
        super(Somoformer, self).__init__()

        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.device = device

        self.d_model = nhid  # hidden size

        # Input and output layers
        self.input_projection = nn.Linear(backcast_size, self.d_model)
        self.output_projection = nn.Linear(self.d_model, forecast_size)

        # Positional Encoding
        self.positional_encoding = TriplePositionalEncoding(
            d_model=nhid,
            feature_types=feature_types,
            n_tickers=n_tickers,
            max_time_steps=max_time_steps,
            dropout=dropout,
            device=device
        )

        # Create a list of encoder layers with alternating attention types
        encoder_layers = []
        attention_types = ['cross', 'self']
        for i in range(nlayers):
            attention_type = attention_types[i % len(attention_types)]
            encoder_layers.append(CustomTransformerEncoderLayer(
                fc_over_bc=forecast_size / backcast_size,
                d_model=nhid,
                nhead=nhead,
                dim_feedforward=dim_feedfwd,
                dropout=dropout,
                activation=activation,
                attention_type=attention_type
            ).to(device))

        self.transformer_layers = nn.Sequential(*encoder_layers)

        self.init_weight_magnitude = init_weight_magnitude
        self.apply(self.init_weights)  # Key to proper initialization

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0, std=self.init_weight_magnitude)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, mean=0, std=self.init_weight_magnitude)

    def forward(self, x, time_indices):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, n_features]
        """
        # Prepare input
        x = x.transpose(0, 1)  # [batch_size, n_features, input_len] -> [n_features, batch_size, input_len]
        x_proj = self.input_projection(x)  # [n_features, batch_size, nhid]

        x_proj = self.positional_encoding(x_proj, time_indices)

        # Pass through transformer layers
        src = self.transformer_layers(x_proj)

        # Extract future predictions
        out_future = src[-self.forecast_size:, :, :]  # [forecast_size, batch_size, nhid]
        out_future = out_future.permute(1, 0, 2)  # [batch_size, forecast_size, nhid]

        # Output projection
        output = self.output_projection(out_future)  # [batch_size, forecast_size, n_features]

        return output

def main(config_path: str = ''):
    config = dynamic_load_config(config_path, Config)

    # Adjust the data loader to provide x of shape [batch_size, seq_len, n_features]
    data_loader, test_loader = get_data_loaders(config.backcast_size, config.forecast_size, test_size_ratio=0.2,
                                                batch_size=config.batch_size, dataset_col=config.test_col)

    model = Somoformer(backcast_size=config.backcast_size,
                       forecast_size=config.forecast_size,
                       nhid=config.nhid,
                       nhead=config.nhead,
                       dim_feedfwd=config.dim_feedfwd,
                       nlayers=config.nlayers,
                       dropout=config.dropout,
                       init_weight_magnitude=config.init_weight_magnitude,
                       device=str(device)).to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    def loss_function(y_pred, y_true):
        return F.mse_loss(y_pred, y_true[...,-config.forecast_size:])

    train_model(model, data_loader, test_loader, loss_function, optimizer, scheduler, config.epochs)

if __name__ == '__main__':
    main()
