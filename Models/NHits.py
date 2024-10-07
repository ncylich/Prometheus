# Reference GitHub: https://github.com/Nixtla/neuralforecast
from torchvision.ops.misc import interpolate

from Train.train import train_model, get_data_loaders
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


lr = 1e-3
batch_size = 1024
epochs = 50
init_weight_magnitude = 1e-3

include_volume = 0
forecast_size = 36
backcast_size = forecast_size * 2

stack_pools = [18, 8, 4, 2, 1]
stack_mlp_freq_downsamples = [36, 24, 8, 2, 1]  # Doubled once

hidden_dim = 256
n_blocks = 6
n_layers = 3
interpolate = 'linear'

test_size_ratio = .2
test_sample_size = 100

test_col = 'close'  # 'y'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NHitsBlock(nn.Module):
    """
    Simple Neural-Hits Block
    Steps:
    1. Max Pool
    2. FFN
    3. BackCast, Forecast Layers
    """
    def __init__(self, backcast_size, forecast_size, n_layers, pool_size, freq_downsample, hidden_dim = 512, interpolate_mode ='linear'):
        super(NHitsBlock, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.hidden_dim = hidden_dim  # backcast_size // pool_size
        self.hidden_downsample = max(math.ceil(self.hidden_dim / freq_downsample), 1)

        self.ffn = nn.ModuleList()
        self.ffn.append(nn.Linear(self.backcast_size // pool_size, self.hidden_dim))
        self.ffn.append(nn.ReLU())
        for _ in range(n_layers-1):
            self.ffn.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.ffn.append(nn.ReLU())
        self.ffn.append(nn.Linear(self.hidden_dim, self.hidden_downsample))
        self.ffn.append(nn.ReLU())

        self.backcast = nn.Linear(self.hidden_downsample, backcast_size // pool_size)  # self.hidden_dim)
        self.forecast = nn.Linear(self.hidden_downsample, forecast_size // pool_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.interpolate_mode = interpolate_mode

    def forward(self, x):
        x = self.pool(x)
        for layer in self.ffn:
            x = layer(x)
        backcast = self.backcast(x)
        forecast = self.forecast(x)

        interpolated_forecast = nn.functional.interpolate(
            forecast.unsqueeze(1),
            size=self.forecast_size,
            mode=self.interpolate_mode
        ).squeeze(1)

        # Expand backcast to match the original input size
        backcast = nn.functional.interpolate(
            backcast.unsqueeze(1),
            size=self.backcast_size,
            mode='nearest'
        ).squeeze(1)

        return backcast, interpolated_forecast

class Nhits(nn.Module):
    """
    Simple Neural Hits model
    Steps:
    1. Make blocks
        A. each block has a pool size and a frequency downsample
    2. Forward pass
        A. MaxPool1D
        B. block
        C. subtract backcast from input, add forecast to output
    """
    def __init__(self, backcast_size, forecast_size, n_layers, stack_pools, stack_mlp_freq_downsamples, n_blocks,
                 hidden_dim = 512, interpolate_mode='linear', volume_included=False):
        super(Nhits, self).__init__()
        backcast_size *= 2 if volume_included else 1
        # forecast_size *= 2 if volume_included else 1
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size  # * (2 if volume_included else 1)
        self.n_layers = n_layers
        self.n_blocks = stack_pools
        self.stack_pools = stack_pools
        self.block_mlp_freq_downsamples = stack_mlp_freq_downsamples

        assert(len(stack_pools) == len(stack_mlp_freq_downsamples))
        self.stacks = nn.ModuleList()
        for pool_size, freq_downsample in zip(stack_pools, stack_mlp_freq_downsamples):
            self.stacks.append(nn.ModuleList(
                [NHitsBlock(backcast_size, forecast_size, n_layers, pool_size, freq_downsample, hidden_dim, interpolate_mode)
                    for _ in range(n_blocks)]
            ))


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
                                    batch_size=batch_size, dataset_col=test_col, include_volume=include_volume)

    model = Nhits(backcast_size, forecast_size, n_layers=n_layers, stack_pools=stack_pools,
                  stack_mlp_freq_downsamples=stack_mlp_freq_downsamples, n_blocks=n_blocks,
                  volume_included=include_volume, interpolate_mode=interpolate).to(device)


    def initialize_weights(module):
        if isinstance(module, nn.Linear):
            # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            torch.nn.init.uniform_(module.weight, a=-init_weight_magnitude, b=init_weight_magnitude)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # model.apply(initialize_weights)

    criterion = torch.nn.L1Loss()
    # criterion = MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1)

    train_model(model, data_loader, test_loader, criterion, optimizer, scheduler, epochs, volume_included=include_volume)
