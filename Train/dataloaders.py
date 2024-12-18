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
    def __init__(self, data, backcast_size, forecast_size, predict_col='close', tickers=None):
        if tickers is None:
            tickers = set([col.split('_')[0] for col in list(data.columns) if '_' in col])
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

        self.velocities = {}
        for ticker in tickers:
            sub_data = torch.from_numpy(data[f'{ticker}_{predict_col}'].to_numpy()).float()
            self.velocities[ticker] = sub_data[1:] / sub_data[:-1]

        self.volumes = {ticker: torch.from_numpy(data[f'{ticker}_volume'].to_numpy()[1:]) for ticker in tickers}

    def __len__(self):
        # Takes first ticker and gets the length of the prices
        return len(self.velocities[next(iter(self.tickers))]) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        x_prices = []
        x_volumes = []
        y_changes = []
        y_std_devs = []

        for ticker in self.tickers:
            veolicty_seq = self.velocities[ticker][idx: idx + self.backcast_size]
            volume_seq = self.volumes[ticker][idx: idx + self.backcast_size]

            final_input_price = self.prices[ticker][idx + self.backcast_size - 1]
            final_output_price = self.prices[ticker][idx + self.backcast_size + self.forecast_size - 1]
            output_change = final_output_price / final_input_price

            # -1 for last input price
            output_prices = self.prices[ticker][idx + self.backcast_size - 1: idx + self.backcast_size + self.forecast_size]
            output_velocities = output_prices[1:] - output_prices[:-1]  # absolute velocity
            output_velocities = output_velocities / final_input_price  # scaled velocity
            std_dev = torch.std(output_velocities)

            x_prices.append(veolicty_seq)
            x_volumes.append(volume_seq)
            y_changes.append(output_change)
            y_std_devs.append(std_dev)

        x_prices = torch.stack(x_prices)  # Shape: [num_tickers, backcast_size]
        x_volumes = torch.stack(x_volumes)  # Shape: [num_tickers, backcast_size]
        x = torch.stack([x_prices, x_volumes])  # Shape: [2, num_tickers, backcast_size]

        y_changes = torch.stack(y_changes)  # Shape: [num_tickers]
        y_std_devs = torch.stack(y_std_devs)  # Shape: [num_tickers]
        y = torch.stack([y_changes, y_std_devs])  # Shape: [2, num_tickers]

        time = torch.stack([self.hours[idx + self.backcast_size],
                            self.month[idx + self.backcast_size],
                            self.year[idx + self.backcast_size]])  # Shape: [3]

        return x.to(device), y.to(device), time.to(device)

def test_train_split(df, test_size_ratio):
    test_len = int(len(df) * test_size_ratio)
    return df.head(len(df) - test_len).copy(), df.tail(test_len).copy()

# mock dataloader
def get_long_term_Xmin_data_loaders(backcast_size, forecast_size, x_min=5, test_size_ratio=.2, batch_size=512,
                                    dataset_col='close', truncate_data=True):
    dataset_path = f'Prometheus/Local_Data/{x_min}min_long_term_merged_UNadjusted.parquet'
    return get_long_term_data_loaders(backcast_size, forecast_size, test_size_ratio, batch_size, dataset_col,
                                      dataset_path, truncate_data)


def get_long_term_data_loaders(backcast_size, forecast_size, test_size_ratio=.2, batch_size=512, dataset_col='close',
                        dataset_path='Prometheus/Local_Data/5min_long_term_merged_UNadjusted.parquet', truncate_data=True):
    path_dirs = os.getcwd().split('/')[::-1]
    try:
        prometheus_idx = path_dirs.index('Prometheus')
    except ValueError:
        prometheus_idx = -1
    dataset_path = '../' * (prometheus_idx + 1) + dataset_path
    data = pd.read_parquet(dataset_path)

    # truncate data to 100k items
    if truncate_data:
        data = data[:100000]

    train_data, test_data = test_train_split(data, test_size_ratio)

    train_dataset = StockDataset(train_data, backcast_size, forecast_size, predict_col=dataset_col)
    test_dataset = StockDataset(test_data, backcast_size, forecast_size, predict_col=dataset_col)

    data_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
    return data_dataloader, test_dataloader

'''
OUTPUT:
100%|██████████| 79989/79989 [00:15<00:00, 5263.17it/s]
Mean Change: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0001, 1.0000, 1.0000])
Mean Std Dev: tensor([0.0016, 0.0022, 0.0021, 0.0004, 0.0012, 0.0050, 0.0028, 0.0005])
100%|██████████| 20/20 [00:03<00:00,  5.82it/s]
Residual Changes: tensor([4.3875e-06, 6.9440e-06, 3.6015e-05, 4.8893e-07, 3.4257e-06, 1.1219e-04,
        3.8588e-05, 7.2155e-07])
Residual Std Devs: tensor([6.6536e-07, 1.7135e-06, 1.7487e-06, 4.1230e-08, 5.5523e-07, 4.1145e-06,
        2.1094e-06, 6.7735e-08])
Changes Loss: tensor(0.0005, dtype=torch.float64)
Std Devs Loss: tensor(2.7539e-05, dtype=torch.float64)
'''
if __name__ == '__main__':
    data_loader, test_loader = get_long_term_Xmin_data_loaders(5, 5, x_min=5, batch_size=1)
    mean_change = torch.zeros(8)
    mean_std_dev = torch.zeros(8)
    for x, y, time in tqdm(data_loader):
        # print(x.shape, y.shape, time.shape)
        mean_change += y[0, 0]
        mean_std_dev += y[0, 1]
    mean_change /= len(data_loader)
    mean_std_dev /= len(data_loader)
    print('Mean Change:', mean_change)
    print('Mean Std Dev:', mean_std_dev)

    residual_changes = torch.zeros(8)
    residual_std_devs = torch.zeros(8)
    changes_loss = 0
    std_devs_loss = 0
    for x, y, time in tqdm(test_loader):
        residual_change = y[0, 0] - mean_change
        residual_std_dev = y[0, 1] - mean_std_dev
        residual_changes += residual_change * residual_change
        residual_std_devs += residual_std_dev * residual_std_dev

        # use MSE loss between predicted and actual using mean change and mean std dev
        changes_loss += F.mse_loss(mean_change, y[0, 0])
        std_devs_loss += F.mse_loss(mean_std_dev, y[0,1])

    residual_changes /= len(test_loader)
    residual_std_devs /= len(test_loader)

    print('Residual Changes:', residual_changes)
    print('Residual Std Devs:', residual_std_devs)
    print('Changes Loss:', changes_loss)
    print('Std Devs Loss:', std_devs_loss)
