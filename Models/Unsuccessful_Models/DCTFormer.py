import sys
if 'google.colab' in sys.modules:
    from Prometheus.Train.train_somoformer import train_model, get_original_data_loaders, get_long_term_data_loaders
    from Prometheus.Models.load_config import dynamic_load_config, update_config_with_factor
else:
    from Train.train_somoformer import train_model, get_long_term_data_loaders
    from Models.Unsuccessful_Models.load_config import dynamic_load_config, update_config_with_factor

from torch.nn import init
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
from dataclasses import dataclass
# import torch_dct as dct
# from utils.dct import get_dct_matrix

# HParams class for DCTFormer model
@dataclass
class Config:
    forecast_size: int = 36
    backcast_size: int = forecast_size * 2

    factor: int = 1
    seq_len: int = backcast_size + forecast_size
    nhid: int = 128 * factor
    nhead: int = 8
    dim_feedfwd: int = 512 * factor
    nlayers: int = 12
    dropout: float = 0.1
    batch_size: int = 1024
    test_col: str = 'close'

    lr: float = 2e-4
    epochs: int = 100
    init_weight_magnitude: float = 1e-2

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


class FeatureTimePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, feature_types: int, max_time_steps: int = 24, dropout: float = 0.1, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        # Embeddings for feature types (e.g., price and volume)
        self.feature_type_encoding = nn.Embedding(feature_types, d_model // 2).to(device)
        # Embeddings for time steps (e.g., hours of the day)
        self.time_encoding = nn.Embedding(max_time_steps, d_model // 2).to(device)

    def forward(self, x: torch.Tensor, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
            time_indices: Tensor of shape [batch_size], indices of time steps
        """
        num_features, batch_size, d_model = x.size()
        half = x.size(2) // 2

        # Add feature type encoding
        x[:, :, 0:half*2:2] = x[:, :, 0:half*2:2] + self.feature_type_encoding(torch.arange(num_features)).to(self.device).unsqueeze(1)

        # Add time encoding
        x[:, :, 1:half*2:2] = x[:, :, 1:half*2:2] + self.time_encoding(time_indices).to(self.device).unsqueeze(0)

        return self.dropout(x)


class TriplePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, feature_types: int, n_tickers: int, max_time_steps: int = 24,
                 dropout: float = 0.1, device='cuda:0'):
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

class DCTFormer(nn.Module):
    def __init__(self, seq_len, forecast_size, nhid=256, nhead=8, dim_feedfwd=1024, nlayers=6, dropout=0.1,
                 activation='gelu', init_weight_magnitude = 1e-2, device='cuda:0', feature_types=2, n_tickers=8, max_time_steps=24, dct_n=108):
        super(DCTFormer, self).__init__()

        self.seq_len = seq_len
        self.forecast_size = forecast_size
        self.device = device
        self.dct_n = dct_n

        # DCT matrices
        dct_m, idct_m = get_dct_matrix(seq_len)
        self.dct = torch.from_numpy(dct_m[:dct_n]).float().to(device)  # [dct_n, seq_len]
        self.idct = torch.from_numpy(idct_m[:, :dct_n]).float().to(device)  # [seq_len, dct_n]

        # Input and output layers
        self.fc_in = nn.Linear(dct_n, nhid)
        self.fc_out = nn.Linear(nhid, dct_n)

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

        self.init_weight_magnitude = init_weight_magnitude
        self.apply(self.init_weights)  # Key to properly working

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # init.xavier_uniform_(m.weight)  # Xavier  # +/- sqrt(features)
            # init.zeros_(m.weight)  # Zeros
            init.normal_(m.weight, mean=0, std=self.init_weight_magnitude)
            if m.bias is not None:
                init.zeros_(m.bias)
                # init.normal_(m.bias, mean=0, std=init_weight_magnitude)
        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, mean=0, std=self.init_weight_magnitude)

    def dct_forward(self, x):
        """
        x: Tensor of shape [batch_size, V, seq_len]
        Returns: Tensor of shape [batch_size, V, dct_n]
        """
        batch_size, V, seq_len = x.size()
        x = x.reshape(-1, seq_len)  # [batch_size * V, seq_len]
        x_dct = self.dct @ x.T  # [dct_n, batch_size * V]
        x_dct = x_dct.T.reshape(batch_size, V, self.dct_n) # [batch_size, V, dct_n]
        return x_dct

    def dct_backward(self, x_dct):
        """
        x_dct: Tensor of shape [batch_size, V, dct_n]
        Returns: Tensor of shape [batch_size, V, seq_len]
        """
        batch_size, V, dct_n = x_dct.size()
        # Reshape x_dct to combine batch_size and V for matrix multiplication
        x_dct = x_dct.reshape(-1, dct_n)  # Shape: [batch_size * V, dct_n]

        # Perform inverse DCT
        # self.idct: [seq_len, dct_n]
        # x_dct.T: [dct_n, batch_size * V]
        x_reconstructed = self.idct @ x_dct.T  # Shape: [seq_len, batch_size * V]

        # Reshape the output back to [batch_size, V, seq_len]
        x_reconstructed = x_reconstructed.T.reshape(batch_size, V, self.seq_len)  # Shape: [batch_size, V, seq_len]
        return x_reconstructed

    def post_process(self, x_dict):
        return self.dct_backward(x_dict)

    def forward(self, x, time_indices):
        # print(x.shape, time_indices.shape)
        batch_size, n_tokens, in_F = x.size() # [batch_size, V, in_F]

        F = self.seq_len
        out_F = F - in_F

        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)
        x = x[:, :, i_idx]
        # print(x.shape)

        # Prepare input
        x = self.dct_forward(x) # [batch_size, V, dct_n]
        x = x.transpose(0, 1)  # [V, batch_size, dct_n]
        x = self.fc_in(x)  # [V, batch_size, nhid]

        # Positional Encoding
        x = self.positional_encoding(x, time_indices)  # [V, batch_size, nhid]

        # Transformer Encoder
        out = self.transformer(x)  # [V, batch_size, nhid]

        # Output layer
        out = self.fc_out(out)  # [V, batch_size, dct_n]

        # Transpose to [batch_size, V, dct_n]
        out = out.permute(1, 0, 2)  # [batch_size, V, dct_n]

        return out

def main(config_path: str = ''):
    config = dynamic_load_config(config_path, Config)
    config = update_config_with_factor(config)

    # data_loader, test_loader = get_original_data_loaders(config.backcast_size, config.forecast_size, test_size_ratio=0.2,
    #                                                      batch_size=config.batch_size, dataset_col=config.test_col)

    data_loader, test_loader = get_long_term_data_loaders(config.backcast_size, config.forecast_size, test_size_ratio=0.2,
                                                          batch_size=config.batch_size, dataset_col=config.test_col)

    model = DCTFormer(config.seq_len,
                      config.forecast_size,
                      nhid=config.nhid,
                      nhead=config.nhead,
                      dim_feedfwd=config.dim_feedfwd,
                      nlayers=config.nlayers,
                      dropout=config.dropout,
                      init_weight_magnitude = config.init_weight_magnitude,
                      device=str(device)).to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    patience = max(1, math.floor(math.log(config.epochs, math.e)))  # floor of ln(epochs) and at least 1
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience)

    def loss_function(y_pred, y_true):
        if config.aux_loss_weight <= 0:
            return F.mse_loss(y_pred, y_true)

        # y_true: [batch_size, V, F]
        # print(y_pred.shape, y_true.shape)
        # recon_velocities = model.dct_backward(y_pred)[..., -forecast_size:]
        # y_forecast = y_true[..., -forecast_size:]
        # calculating difference in summed_velocities
        summed_true = y_true[..., -config.forecast_size:].sum(dim=-1)
        summed_pred = y_pred[..., -config.forecast_size:].sum(dim=-1)

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

        return (1 - config.aux_loss_weight) * F.mse_loss(y_pred, y_true) + config.aux_loss_weight * diff_aux_loss #+ 0.2 * zero_dist_aux_loss #+ 0.3 * F.mse_loss(recon_velocities, y_forecast)

    train_model(model, data_loader, test_loader, loss_function, optimizer, scheduler, config.epochs)


if __name__ == '__main__':
    main('../configs/dct_config.yaml')
