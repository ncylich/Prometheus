if __name__ == '__main__':
    from DataCollection.data_processing import read_processed_parquet, test_train_split, read_parquet_nixtla
else:
    from Prometheus.DataCollection.data_processing import read_processed_parquet, test_train_split, read_parquet_nixtla

import numpy as np
from time import sleep
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrudeClosingAndVolumeDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, predict_col='close', mean_std_path=None):
        self.price_data = torch.from_numpy(data[predict_col].to_numpy()).float()
        self.price_velocity = self.price_data[1:] - self.price_data[:-1]  # Shape: [seq_len - 1]
        self.price_velocity = torch.cat([self.price_velocity[0].unsqueeze(0), self.price_velocity])  # Shape: [seq_len]
        self.volume_data = torch.from_numpy(data['volume'].to_numpy()).float() / 1e3  # DOWN-SCALING
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.times = torch.tensor(data['date'].dt.hour.values)

        # Compute mean and std if not provided
        if mean_std_path and os.path.exists(mean_std_path):
            # Load precomputed mean and std
            mean_std = torch.load(mean_std_path)
            self.price_mean = mean_std['price_mean']
            self.price_std = mean_std['price_std']
            self.volume_mean = mean_std['volume_mean']
            self.volume_std = mean_std['volume_std']
        else:
            # Compute mean and std
            self.price_mean = self.price_data.mean()
            self.price_std = self.price_data.std()
            self.volume_mean = self.volume_data.mean()
            self.volume_std = self.volume_data.std()
            # Save mean and std
            if mean_std_path:
                mean_std = {
                    'price_mean': self.price_mean,
                    'price_std': self.price_std,
                    'volume_mean': self.volume_mean,
                    'volume_std': self.volume_std
                }
                torch.save(mean_std, mean_std_path)

    def __len__(self):
        return len(self.price_data) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        # Get sequences
        price_seq = self.price_data[idx: idx + self.backcast_size + self.forecast_size]
        volume_seq = self.volume_data[idx: idx + self.backcast_size + self.forecast_size]

        price_velocity = self.price_velocity[idx: idx + self.backcast_size + self.forecast_size]

        # Normalize sequences
        # price_seq = (price_seq - self.price_mean) / self.price_std
        volume_seq = (volume_seq - self.volume_mean) / self.volume_std

        # Prepare input and output sequences
        x_price = price_velocity[:self.backcast_size]
        x_volume = volume_seq[:self.backcast_size]
        y_price = price_velocity
        y_volume = volume_seq

        # Combine price and volume
        x = torch.stack([x_price, x_volume])  # Shape: [2, backcast_size]
        y = torch.stack([y_price, y_volume])  # Shape: [2, backcast_size + forecast_size]

        time = self.times[idx + self.backcast_size]

        return x.to(device), y.to(device), time.to(device), price_seq.to(device)


class MultiStockClosingAndVolumeDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, predict_col='close', tickers=None):
        # columns in data csv: ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        if tickers is None:
            tickers = ['CL', 'GC', 'NG', 'ES', 'ZN', 'DX', 'HG']

        data['date'] = pd.to_datetime(data['date'], errors='coerce', utc=True)

        self.tickers = tickers
        self.velocities = {ticker: self.calculate_velocity(torch.from_numpy(data[ticker + '_' + predict_col].to_numpy()).float()) for ticker in tickers}
        self.volumes = {ticker: torch.from_numpy(data[ticker + '_volume'].to_numpy()).float() / 1e3 for ticker in tickers}
        self.price_data = torch.from_numpy(data['CL_' + predict_col].to_numpy()).float()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.times = torch.tensor(data['date'].dt.hour.values)

    def __len__(self):
        return len(self.price_data) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        x_prices = []
        x_volumes = []
        y_prices = []
        y_volumes = []
        CL_price_seq = self.price_data[idx: idx + self.backcast_size + self.forecast_size]

        for ticker in self.tickers:
            price_seq = self.velocities[ticker][idx: idx + self.backcast_size + self.forecast_size]
            volume_seq = self.volumes[ticker][idx: idx + self.backcast_size + self.forecast_size]

            x_price = price_seq[:self.backcast_size]
            x_volume = volume_seq[:self.backcast_size]
            y_price = price_seq
            y_volume = volume_seq

            x_prices.append(x_price)
            x_volumes.append(x_volume)
            y_prices.append(y_price)
            y_volumes.append(y_volume)

        x = torch.stack(x_prices + x_volumes)  # Shape: [2, num_tickers, backcast_size]
        y = torch.stack(y_prices + y_volumes)  # Shape: [2, num_tickers, backcast_size + forecast_size]

        time = self.times[idx + self.backcast_size]

        return x.to(device), y.to(device), time.to(device), CL_price_seq.to(device)
    def calculate_velocity(self, data):
        velocity = data[1:] - data[:-1]
        return torch.cat([velocity[0].unsqueeze(0), velocity])

def plot_forecast_vs_actual(forecast, actual, gt_seq):
    plt.figure(figsize=(12, 6))
    plt.plot(forecast, label='Forecast')
    plt.plot(actual, label='Actual')
    # plt.plot(gt_seq, label='Ground Truth')
    plt.legend()
    plt.show()


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    model.train()

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for x, y, t, gt_seq in train_loader:
            optimizer.zero_grad()
            forecast = model(x, t)
            loss = criterion(forecast, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')
        scheduler.step(epoch_loss)

        test_losses = torch.tensor([0,0], dtype=torch.float32)
        model.eval()
        total_correct_ups = 0
        total_correct_downs = 0
        total_correct_overall = 0
        with torch.no_grad():
            for x, y, t, gt_seq in test_loader:
                output = model(x, t)
                try:
                    forecast = model.dct_backward(output) # [batch_size, V, seq_len]
                except AttributeError:
                    forecast = output
                # halving values to only consider price data (if applicable)
                # print(y.shape)
                # plot_forecast_vs_actual(forecast[0], y[0])
                forecast = forecast[:, 0].squeeze(1)
                y = y[:, 0].squeeze(1)
                torch.cumsum(forecast, dim=-1, out=forecast)
                torch.cumsum(y, dim=-1, out=y)
                # print(forecast.shape, y.shape, gt_seq[:, 0].shape)
                forecast += gt_seq[:, 0].unsqueeze(1)
                y += gt_seq[:, 0].unsqueeze(1)

                forecast += y[:, x.size()[-1]].unsqueeze(1) - forecast[:, x.size()[-1]].unsqueeze(1)

                plot_forecast_vs_actual(forecast[0], y[0], gt_seq[0])

                # calculate percentage of correct ups, correct downs, and correct overall
                sign_truth = torch.sign(y[:, -1] - y[:, x.size()[-1]])
                sign_forecast = torch.sign(forecast[:, -1] - forecast[:, x.size()[-1]])
                correct_ups = torch.sum((sign_truth + sign_forecast) == 2) / torch.sum(sign_truth == 1)
                correct_downs = torch.sum((sign_truth + sign_forecast) == -2) / torch.sum(sign_truth == -1)
                correct_overall = torch.sum(sign_truth == sign_forecast) / len(sign_truth)
                total_correct_ups += correct_ups
                total_correct_downs += correct_downs
                total_correct_overall += correct_overall

                forecast = forecast[:, -model.forecast_size:]
                y = y[:, -model.forecast_size:]


                test_losses += mae_and_mse_loss(forecast, y)
        test_losses /= len(test_loader)
        sleep(1e-5)
        print(f'Test MAE Loss: {test_losses[0]}, MSE Loss: {test_losses[1]}')
        print(f'Correct Ups: {total_correct_ups/len(test_loader)}, Correct Downs: {total_correct_downs/len(test_loader)}, Correct Overall: {total_correct_overall/len(test_loader)}')
        sleep(1e-5)


def mae_and_mse_loss(forecast, actual):
    return torch.tensor([F.l1_loss(forecast, actual), F.mse_loss(forecast, actual)])


def get_data_loaders(backcast_size, forecast_size, test_size_ratio=.2, batch_size=512,
                     dataset_path='Prometheus/DataCollection/20241030_merged_squeezed.csv', dataset_col='close'):

    # Updating data path dynamically on dir
    path_dirs = os.getcwd().split('/')[::-1]
    try:
        prometheus_idx = path_dirs.index('Prometheus')
    except ValueError:
        prometheus_idx = -1
    dataset_path = '../' * (prometheus_idx + 1) + dataset_path

    data = pd.read_csv(dataset_path)
    data = data[data['expiry-dist'] == 3].copy()  # Only considering 3-month futures
    train_data, test_data = test_train_split(data, test_size_ratio)

    mean_std_path = 'mean_std.pth'  # Path to save/load mean and std

    CrudeDataset = MultiStockClosingAndVolumeDataset
    train_dataset = CrudeDataset(train_data, backcast_size, forecast_size, predict_col=dataset_col)
    test_dataset = CrudeDataset(test_data, backcast_size, forecast_size, predict_col=dataset_col)
    data_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024)
    return data_dataloader, test_dataloader


if __name__ == '__main__':
    data_loader, test_loader = get_data_loaders(72, 36, test_size_ratio=.2, batch_size=128)