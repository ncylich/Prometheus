from torchvision.ops.misc import interpolate
import sys
if 'google.colab' in sys.modules:
    from Prometheus.Train.train_somoformer import train_model_split, get_data_loaders
else:
    from Train.train_somoformer import train_model_split, get_data_loaders

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
# import torch_dct as dct
# from utils.dct import get_dct_matrix


# Parameters
# TODO: Enlarge Model and FT Params for it

forecast_size = 36
backcast_size = forecast_size * 2

factor = 2
seq_len = backcast_size + forecast_size
nhid = 128 * factor
nhead = 8
dim_feedfwd = 512 * factor
nlayers = 12
dropout = 0.1
batch_size = 1024
test_col = 'close'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr1 = 1e-3
lr2 = 1e-4
epochs = 100
init_weight_magnitude = 1e-2 #/ (factor ** 2)

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

def init_weights(m):
    if isinstance(m, nn.Linear):
        # init.xavier_uniform_(m.weight)  # Xavier (+/- sqrt(features))
        # init.zeros_(m.weight)  # Zeros
        init.normal_(m.weight, mean=0, std=init_weight_magnitude)
        if m.bias is not None:
            init.zeros_(m.bias)
            # init.normal_(m.bias, mean=0, std=init_weight_magnitude)
    elif isinstance(m, nn.Embedding):
        init.normal_(m.weight, mean=0, std=init_weight_magnitude)

class Somoformer(nn.Module):
    def __init__(self, backcast_size, forecast_size, nhid=256, nhead=8, dim_feedfwd=1024, nlayers=6,
                     dropout=0.1, activation='gelu', device='cuda:0', feature_types=2, n_tickers=7, max_time_steps=24):
        super(Somoformer, self).__init__()

        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.device = device

        # Input and output layers
        self.fc_in = nn.Linear(backcast_size, nhid)
        self.fc_out = nn.Linear(nhid, forecast_size)
        #self.fc_out = nn.Sequential(nn.Linear(nhid, nhid), nn.Sigmoid(), nn.Linear(nhid, seq_len))

        # Positional Encoding
        self.positional_encoding = TriplePositionalEncoding(
            d_model=nhid,
            feature_types=feature_types,
            n_tickers=n_tickers,
            max_time_steps=max_time_steps,
            dropout=dropout,
            device=device
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.apply(init_weights)  # Key to properly working

    def post_process(self, x):
        return x

    @staticmethod
    def fix_mean_offset(x, out):
        input_size = x.size(-1)
        output_size = out.size(-1)
        x_means = x.mean(dim=-1, keepdim=True)
        x_means = x_means.permute(1, 0, 2)
        out_input_means = out[..., :input_size].mean(dim=-1, keepdim=True)
        delta_means = x_means - out_input_means
        delta_means = delta_means.repeat(1, 1, output_size)
        out = out + delta_means
        return out

    def forward(self, x, time_indices):
        # print(x.shape, time_indices.shape)
        # print(x.shape)

        # Prepare input
        x = x.transpose(0, 1)  # [V, batch_size, seq_len]
        x = self.fc_in(x)  # [V, batch_size, nhid]

        # Positional Encoding
        x = self.positional_encoding(x, time_indices)  # [V, batch_size, nhid]

        # Transformer Encoder
        out = self.transformer(x)  # [V, batch_size, nhid]

        # Output layer
        out = self.fc_out(out)  # [V, batch_size, seq_len]

        # Transpose to [batch_size, V, seq_len]
        out = out.permute(1, 0, 2)  # [batch_size, V, seq_len]

        # Post-process correction - artificially, makes better, but not good enough
        # out = Somoformer.fix_mean_offset(x, out)

        return out

def main(lr=lr1, w1=0.5, w2=0.5):
    data_loader, test_loader = get_data_loaders(backcast_size, forecast_size, test_size_ratio=0.2,
                                                batch_size=batch_size, dataset_col=test_col)

    model = Somoformer(backcast_size,
                       forecast_size,
                       nhid=nhid,
                       nhead=nhead,
                       dim_feedfwd=dim_feedfwd,
                       nlayers=nlayers,
                       dropout=dropout,
                       device=device).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    def loss_function(y_pred, y_true):
        # y_true: [batch_size, V, F]
        # print(y_pred.shape, y_true.shape)
        # recon_velocities = model.dct_backward(y_pred)[..., -forecast_size:]
        # y_forecast = y_true[..., -forecast_size:]
        # calculating difference in summed_velocities
        y_true = y_true[..., -forecast_size:]
        summed_true = y_true.sum(dim=-1)
        summed_pred = y_pred.sum(dim=-1)

        #full_sum = y_pred.sum(dim=-1)
        #Zero_sum = torch.zeros_like(full_sum)

        # squared difference in sigmoid
        diff_aux_loss = F.mse_loss(torch.sigmoid(summed_pred), torch.sigmoid(summed_true))

        #zero_dist_aux_loss = F.mse_loss(full_sum, Zero_sum)

        # plt.figure(figsize=(9, 6))
        # plt.plot(y_pred[0][0].clone().detach().cpu(), label='Forecast')
        # plt.plot(y_true[0][0].clone().detach().cpu(), label='Actual')
        # plt.legend()
        # plt.show()

        return w1 * F.mse_loss(y_pred, y_true) + w2 * diff_aux_loss #+ 0.2 * zero_dist_aux_loss #+ 0.3 * F.mse_loss(recon_velocities, y_forecast)

    train_model_split(model, data_loader, test_loader, loss_function, optimizer, scheduler, epochs)


if __name__ == '__main__':
    main(lr1, 1, 0)
    # main(lr1, 0.5, 0.5)
    # main(lr2)