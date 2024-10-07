from DataCollection.data_processing import read_processed_parquet, test_train_split, read_parquet_nixtla
from Train.train import train_model, get_data_loaders
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
from enum import Enum

class BlockType(Enum):
    IDENTITY = 1
    TREND = 2
    SEASONALITY = 3

lr = 1e-3
batch_size = 512
epochs = 50
dropout = 0.0

forecast_size = 36
backcast_size = forecast_size * 2

n_harmonics = 3
poly_degree = 3
n_blocks = 3
stacks = [BlockType.IDENTITY, BlockType.TREND, BlockType.SEASONALITY]
# stack_random_weights = [1e-1, 10 ** (-poly_degree), 1e-1]
# init_weight_magnitude = 1e-3
hidden_dim = 512

test_size_ratio = .2
test_sample_size = 100
test_col = 'close'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NBeatsBlock(nn.Module):
    """
    Simple Neural-Beats Block
    Steps:
    1. FFN
    2. BackCast, Forecast Layers
    """
    def __init__(self, backcast_size, forecast_size, n_layers=4, block_type = BlockType.IDENTITY, n_harmonics=2,
        poly_degree=2, hidden_dim=-1, dropout=0.2):
        # poly degree must be small (<5)
        super(NBeatsBlock, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.hidden_dim = (backcast_size + forecast_size) * 4 if hidden_dim == -1 else hidden_dim
        self.block_type = block_type
        self.n_harmonics = n_harmonics
        self.poly_degree = poly_degree
        self.dropout=dropout

        self.ffn = nn.ModuleList()
        start_size = self.backcast_size
        for _ in range(n_layers):
            self.ffn.append(nn.Linear(start_size, self.hidden_dim))
            start_size = self.hidden_dim
            self.ffn.append(nn.ReLU())  # GELU?
            self.ffn.append(nn.Dropout(dropout))

        if block_type == block_type.IDENTITY:
            self.final_layer_output_size = -1
            self.backcast = nn.Linear(self.hidden_dim, backcast_size)
            self.forecast = nn.Linear(self.hidden_dim, forecast_size)

            self.init_weight_magnitude = 1e-1
            self.initialize_ffn_weights()
        else:
            self.final_layer_output_size = 2 * n_harmonics if block_type == block_type.SEASONALITY else poly_degree
            self.backcast = nn.Linear(self.hidden_dim, self.final_layer_output_size)
            self.forecast = nn.Linear(self.hidden_dim, self.final_layer_output_size)

            if block_type == block_type.SEASONALITY:
                self.init_weight_magnitude = 1e-1
                self.initialize_ffn_weights()
                # Create time grids for backcast and forecast
                time_backcast = torch.linspace(-1, 0, self.backcast_size).unsqueeze(0)  # Backcast time grid
                time_forecast = torch.linspace(0, 1, self.forecast_size).unsqueeze(0)  # Forecast time grid

                # Compute sine and cosine basis
                self.register_buffer('fourier_backcast', self._compute_fourier_basis(time_backcast))
                self.register_buffer('fourier_forecast', self._compute_fourier_basis(time_forecast))
            elif block_type == block_type.TREND:
                self.init_weight_magnitude = 10 ** (-poly_degree)
                self.initialize_ffn_weights()
                # Poly matrices
                poly_backcast = [[math.pow(n, i) for n in range(self.backcast_size, 0, -1)] for i in range(self.poly_degree)]
                poly_forecast = [[math.pow(n, i) for n in range(1, self.forecast_size + 1)] for i in range(self.poly_degree)]

                # Convert them to tensors and set them as parameters or buffers (this way they're part of the model)
                self.register_buffer('poly_backcast', torch.tensor(poly_backcast, dtype=torch.float32))
                self.register_buffer('poly_forecast', torch.tensor(poly_forecast, dtype=torch.float32))

    def _compute_fourier_basis(self, time_grid):
        """
        Generate the Fourier basis matrix (sine and cosine).
        """
        basis = []
        for k in range(1, self.n_harmonics + 1):
            basis.append(torch.sin(2 * math.pi * k * time_grid))
            basis.append(torch.cos(2 * math.pi * k * time_grid))
        return torch.cat(basis, dim=0)  # Shape: (2 * n_harmonics, time_grid_size)

    def forward(self, x):
        for layer in self.ffn:
            x = layer(x)
        backcast = self.backcast(x)
        forecast = self.forecast(x)

        if self.block_type == BlockType.SEASONALITY:
            backcast = torch.matmul(backcast, self.fourier_backcast)
            forecast = torch.matmul(forecast, self.fourier_forecast)
        elif self.block_type == BlockType.TREND:
            backcast = torch.matmul(backcast, self.poly_backcast)
            forecast = torch.matmul(forecast, self.poly_forecast)

        return backcast, forecast

    def initialize_ffn_weights(self):
        def initialize_weights(module):
            if isinstance(module, nn.Linear):
                # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                torch.nn.init.uniform_(module.weight, a=-self.init_weight_magnitude, b=self.init_weight_magnitude)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.ffn.apply(initialize_weights)


class NBeats(nn.Module):
    def __init__(self, backcast_size, forecast_size, n_blocks, stacks, hidden_dim=-1,
                 n_harmonics=2, poly_degree=2, dropout=0.0, volume_included=False):
        super(NBeats, self).__init__()
        self.backcast_size = backcast_size * (2 if volume_included else 1)
        self.forecast_size = forecast_size
        self.n_blocks = n_blocks
        self.stacks = stacks

        self.stacks = nn.ModuleList()
        for btype in stacks:
            stack = nn.ModuleList()
            for _ in range(n_blocks):
                stack.append(NBeatsBlock(backcast_size, forecast_size, block_type=btype, hidden_dim=hidden_dim,
                                         n_harmonics=n_harmonics, poly_degree=poly_degree, dropout=dropout))
            self.stacks.append(stack)

    def forward(self, x):
        net_forecast = torch.zeros_like(x[:, :self.forecast_size])
        for stack in self.stacks:
            for block in stack:
                backcast, forecast = block(x)
                x = x - backcast
                net_forecast = net_forecast + forecast
        return net_forecast


if __name__ == '__main__':
    data_loader, test_loader = get_data_loaders(backcast_size, forecast_size, test_size_ratio=test_size_ratio,
                                                batch_size=batch_size, dataset_col=test_col)

    model = NBeats(backcast_size=backcast_size, forecast_size=forecast_size, n_blocks=n_blocks, stacks=stacks,
                   hidden_dim=hidden_dim, n_harmonics=n_harmonics, poly_degree=poly_degree, dropout=dropout).to(device)

    criterion = torch.nn.L1Loss()
    # criterion = MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
    # step down LR after so many steps

    train_model(model, data_loader, test_loader, criterion, optimizer, scheduler, epochs)
