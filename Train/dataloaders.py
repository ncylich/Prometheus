import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"  #\
    # else "mps" if torch.backends.mps.is_available() else "cpu"

class StockDataset(Dataset):
    ticker_velocity_scale_params = {}
    ticker_volume_scale_params = {}


    def __init__(self, data, backcast_size, forecast_size, training_set, predict_col='close', tickers=None, log_vols=False, use_velocity=False):
        if tickers is None:
            tickers = []
            for ticker in [col.split('_')[0] for col in list(data.columns) if '_' in col]:
                if ticker not in tickers:
                    tickers.append(ticker)

        data['date'] = pd.to_datetime(data['date'], errors='coerce', utc=True)
        data['date'] = data['date'].dt.tz_convert('America/New_York')

        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

        # Off by 1 bc of the removal of first row
        self.hours = torch.tensor(data['date'].dt.hour.values)[1:]
        self.month = torch.tensor(data['date'].dt.month.values)[1:]
        self.year = torch.tensor(data['date'].dt.year.values)[1:]

        self.tickers = tickers
        self.prices = {ticker: torch.from_numpy(data[f'{ticker}_{predict_col}'].to_numpy()[1:]) for ticker in tickers}

        self.prices = {}
        for ticker in tickers:
            prices = data[f'{ticker}_{predict_col}'].to_numpy()
            prices = prices.pct_change()[1:] if use_velocity else prices[1:]
            assert not np.isnan(prices).any()

            if training_set:
                # offset, scale = self.min_max_scale(prices)
                offset, scale = self.mean_std_scale(prices)
                self.ticker_velocity_scale_params[ticker] = (offset, scale)

            offset, scale = self.ticker_velocity_scale_params[ticker]
            prices = (prices - offset) / scale

            assert not np.isnan(prices).any()

            self.prices[ticker] = torch.from_numpy(prices).float()
            # # interpolate all nans and infs
            # self.velocities[ticker][torch.isnan(self.velocities[ticker])] = 1
            # self.velocities[ticker][torch.isinf(self.velocities[ticker])] = 1

        self.volumes = {}
        for ticker in tickers:
            volume = data[f'{ticker}_volume'].to_numpy()[1:]
            if log_vols:
                volume = np.log(volume + 1)

            if training_set:
                offset, scale = self.min_max_scale(volume)
                self.ticker_volume_scale_params[ticker] = (offset, scale)

            offset, scale = self.ticker_volume_scale_params[ticker]
            volume = (volume - offset) / scale

            volume = torch.from_numpy(volume).float()
            self.volumes[ticker] = volume

    def __len__(self):
        # Takes first ticker and gets the length of the prices
        return len(self.prices[next(iter(self.tickers))]) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        x_prices = []
        x_volumes = []
        y_changes = []
        y_stds = []

        for ticker in self.tickers:
            veolicty_seq = self.prices[ticker][idx: idx + self.backcast_size]
            volume_seq = self.volumes[ticker][idx: idx + self.backcast_size]

            final_input_price = self.prices[ticker][idx + self.backcast_size - 1]
            final_output_price = self.prices[ticker][idx + self.backcast_size + self.forecast_size - 1]
            output_change = final_output_price / final_input_price

            # -1 for last input price
            output_prices = self.prices[ticker][idx + self.backcast_size - 1: idx + self.backcast_size + self.forecast_size]
            output_velocities = output_prices[1:] - output_prices[:-1]  # absolute velocity
            output_velocities = output_velocities / final_input_price  # scaled velocity
            std = torch.std(output_velocities)

            x_prices.append(veolicty_seq)
            x_volumes.append(volume_seq)
            y_changes.append(output_change)
            y_stds.append(std)

        x_prices = torch.stack(x_prices)  # Shape: [num_tickers, backcast_size]
        x_volumes = torch.stack(x_volumes)  # Shape: [num_tickers, backcast_size]
        x = torch.stack([x_prices, x_volumes])  # Shape: [2, num_tickers, backcast_size]

        y_changes = torch.stack(y_changes)  # Shape: [num_tickers]
        y_stds = torch.stack(y_stds)  # Shape: [num_tickers]
        y = torch.stack([y_changes, y_stds])  # Shape: [2, num_tickers]

        time = torch.stack([self.hours[idx + self.backcast_size],
                            self.month[idx + self.backcast_size],
                            self.year[idx + self.backcast_size]])  # Shape: [3]

        return x.to(device), y.to(device), time.to(device)

    @staticmethod
    def min_max_scale(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return min_val, max_val - min_val

    @staticmethod
    def mean_std_scale(data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std

class TokenStockDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, token_len, training_set, predict_col='close', tickers=None, log_vols=False):
        assert backcast_size % token_len == 0, f'Backcast size must be divisible by token_len: {backcast_size} % {token_len} != 0'
        self.stock_dataset = StockDataset(data, token_len, forecast_size, training_set, predict_col, tickers, log_vols)
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.num_tokens = backcast_size // token_len
        self.token_len = token_len

    def __len__(self):
        return len(self.stock_dataset.prices[self.stock_dataset.tickers[0]]) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        xs = []
        times = []
        y = None
        for i in range(self.num_tokens):
            x, y, time = self.stock_dataset[idx + i * self.token_len]
            xs.append(x)
            times.append(time)

        x = torch.stack(xs)  # Shape: [num_tokens, 2, num_tickers, token_len]
        x = x.permute(1, 2, 0, 3)  # Shape: [2, num_tickers, num_tokens, token_len]
        x = x.reshape(2, -1, self.token_len)  # Shape: [2, num_tickers * num_tokens, token_len]

        time = torch.stack(times)  # Shape: [num_tokens, 3]

        return x, y, time

def test_train_split(df, test_size_ratio=.2):
    test_len = int(len(df) * test_size_ratio)
    return df.head(len(df) - test_len).copy(), df.tail(test_len).copy()

# mock dataloader
def get_long_term_Xmin_data_loaders(backcast_size, forecast_size, token_len, x_min=5, test_size_ratio=.2,
                                    batch_size=512, dataset_col='close', head_prop = .1):
    dataset_path = f'Prometheus/Local_Data/{x_min}min_long_term_merged_UNadjusted.parquet'
    return get_long_term_data_loaders(backcast_size, forecast_size, token_len, test_size_ratio, batch_size, dataset_col,
                                      dataset_path, head_prop)

def get_long_term_data_loaders(backcast_size, forecast_size, token_len, test_size_ratio=.2, batch_size=512,
                               dataset_col='close', dataset_path='Prometheus/Local_Data/5min_long_term_merged_UNadjusted.parquet', head_prop=.1, ):

    path_dirs = os.getcwd().split('/')[::-1]
    try:
        prometheus_idx = path_dirs.index('Prometheus')
    except ValueError:
        prometheus_idx = -1
    dataset_path = '../' * (prometheus_idx + 1) + dataset_path
    data = pd.read_parquet(dataset_path)

    data = data.head(int(len(data) * head_prop))

    train_data, test_data = test_train_split(data, test_size_ratio)

    if token_len:
        train_dataset = TokenStockDataset(train_data, backcast_size, forecast_size, token_len, True, predict_col=dataset_col)
        test_dataset = TokenStockDataset(test_data, backcast_size, forecast_size, token_len, False, predict_col=dataset_col)
    else:
        train_dataset = StockDataset(train_data, backcast_size, forecast_size, True, predict_col=dataset_col)
        test_dataset = StockDataset(test_data, backcast_size, forecast_size, False, predict_col=dataset_col)

    data_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2*batch_size, shuffle=True)
    return data_dataloader, test_dataloader

'''
OUTPUT:
100%|██████████| 79962/79962 [00:15<00:00, 5235.90it/s]
Mean Change: tensor([0.9999, 1.0000, 1.0002, 1.0001, 1.0006, 0.9998, 1.0000, 1.0002])
Mean Std: tensor([0.0024, 0.0005, 0.0026, 0.0006, 0.0059, 0.0033, 0.0020, 0.0014])
100%|██████████| 20/20 [00:03<00:00,  6.35it/s]
Residual Changes: tensor([2.0759e-04, 2.7756e-06, 3.2590e-05, 9.2425e-06, 5.2072e-04, 2.6806e-04,
        7.9735e-05, 2.4252e-05])
Residual Stds: tensor([7.1884e-07, 4.8953e-08, 1.7324e-06, 8.6941e-08, 3.9700e-06, 1.7752e-06,
        9.4636e-07, 4.7954e-07])
Changes Loss: tensor(0.0029, dtype=torch.float64)
Std Loss: tensor(2.4396e-05, dtype=torch.float64)
'''
def test_main():
    data_loader, test_loader = get_long_term_Xmin_data_loaders(1, 36, x_min=5, batch_size=1)
    mean_change = torch.zeros(8)
    mean_std = torch.zeros(8)
    for x, y, time in tqdm(data_loader):
        mean_change += y[0, 0]
        mean_std += y[0, 1]
    mean_change /= len(data_loader)
    mean_std /= len(data_loader)
    print('Mean Change:', mean_change)
    print('Mean Std:', mean_std)

    residual_changes = torch.zeros(8)
    residual_stds = torch.zeros(8)
    changes_loss = 0
    stds_loss = 0
    for x, y, time in tqdm(test_loader):
        residual_change = y[0, 0] - mean_change
        residual_std = y[0, 1] - mean_std
        residual_changes += residual_change * residual_change
        residual_stds += residual_std * residual_std

        # use MSE loss between predicted and actual using mean change and mean std dev
        changes_loss += F.mse_loss(mean_change, y[0, 0])
        stds_loss += F.mse_loss(mean_std, y[0,1])

    residual_changes /= len(test_loader)
    residual_stds /= len(test_loader)

    print('Residual Changes:', residual_changes)
    print('Residual Stds:', residual_stds)
    print('Changes Loss:', changes_loss)
    print('Std Loss:', stds_loss)

if __name__ == '__main__':
    test_main()
