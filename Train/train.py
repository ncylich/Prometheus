from DataCollection.Old_Nixtla_Methods.data_processing import read_processed_parquet, test_train_split
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.close('all')


class CrudeClosingPriceDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, predict_col='close'):
        self.data = data[predict_col].to_numpy()
        self.time = data['date'].to_numpy()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.data) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.backcast_size]
        y = self.data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size]
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device), idx


class CrudeClosingVelocityDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, predict_col='close'):
        predict_data = data[predict_col].to_numpy()
        self.data = predict_data[1:] - predict_data[:-1]
        self.time = data['date'].to_numpy()[1:]
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    # take away an additional index for the velocity
    def __len__(self):
        return len(self.data) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.backcast_size]
        y = self.data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size]
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device), idx


class CrudeClosingAndVolumeDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, predict_col='close'):
        self.price_data = data[predict_col].to_numpy()
        self.volume_data = data['volume'].to_numpy() / 1e3  # DOWN-SCALING so that Loss is dominated by price
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.price_data) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        x = np.concatenate([self.price_data[idx:idx + self.backcast_size],
                            self.volume_data[idx:idx + self.backcast_size]])
        # y = np.concatenate([self.price_data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size],
        #                     self.volume_data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size]])
        y = np.concatenate([self.price_data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size],
                            self.volume_data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size]])
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device), idx


class CrudeClosingAndVolumeVelocityDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, predict_col='close'):
        price_data = data[predict_col].to_numpy()
        volume_data = data['volume'].to_numpy() / 1e3  # DOWN-SCALING so that Loss is dominated by price

        self.price_data = price_data[1:] - price_data[:-1]
        self.volume_data = volume_data[1:] - volume_data[:-1]
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.price_data) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        x = np.concatenate([self.price_data[idx:idx + self.backcast_size],
                            self.volume_data[idx:idx + self.backcast_size]])
        # y = np.concatenate([self.price_data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size],
        #                     self.volume_data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size]])
        y = self.price_data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size]
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device), idx



def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, volume_included=False, velocity_dataset=False):
    model.train()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for x, y, idx in train_loader:
            optimizer.zero_grad()
            forecast = model(x)
            loss = criterion(forecast, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')
        scheduler.step(epoch_loss)

        test_losses = torch.tensor([0,0], dtype=torch.float32)
        plt.close('all')  # closing all previous plots
        model.eval()
        with torch.no_grad():
            for x, y, idx in test_loader:
                forecast = model(x).squeeze(0)
                # forecast = torch.zeros_like(y)  # create demo horizontal line forecast -> MAE = 0.253, MSE = 0.131

                if volume_included:# halving values to only consider price data (if applicable)
                    forecast = forecast[:, : model.forecast_size]
                    y = y[:, : model.forecast_size]

                if velocity_dataset:
                    x, forecast, y = interpolate_velocities(x, forecast, y)

                test_losses += mae_and_mse_loss(forecast, y)
                # plot_forecasts(x, y, forecast)
                plot_time_forecasts(x, y, forecast, idx, test_loader.dataset)
        test_losses /= len(test_loader)
        sleep(1e-5)
        print(f'Test MAE Loss: {test_losses[0]}, MSE Loss: {test_losses[1]}')
        sleep(1e-5)

def interpolate_velocities(input_data, forecast, actual):
    input_data = input_data.cumsum(dim=-1)
    B, L = actual.size()
    starting_point = input_data[:, -1].unsqueeze(1).expand(B, L)
    forecast = forecast.cumsum(dim=-1) + starting_point
    actual = actual.cumsum(dim=-1) + starting_point
    return input_data, forecast, actual

def mae_and_mse_loss(forecast, actual):
    return torch.tensor([F.l1_loss(forecast, actual), F.mse_loss(forecast, actual)])

def plot_forecasts(input_data, actual, forecasts):
    input_data, actual, forecasts = input_data[0, :].cpu().numpy(), actual[0, :].cpu().numpy(), forecasts[0, :].cpu().numpy()
    input_x = np.arange(len(input_data))
    plt.plot(input_x, input_data, label='Input')

    output_x = np.arange(len(input_data), len(input_data) + len(forecasts))
    plt.plot(output_x, actual, label='Actual')
    plt.plot(output_x, forecasts, label='Forecast', linestyle='--')

    plt.legend()
    plt.show()

def plot_time_forecasts(input_data, actual, forecasts, idxs, dataset):
    input_data, actual, forecasts = input_data[0, :].cpu().numpy(), actual[0, :].cpu().numpy(), forecasts[0, :].cpu().numpy()
    idx = idxs[0].item()
    input_x = dataset.time[idx:idx + len(input_data)]
    output_x = dataset.time[idx + len(input_data):idx + len(input_data) + len(forecasts)]

    # plt.plot(input_x, input_data, label='Input')
    plt.plot(output_x, actual, label='Actual')
    plt.plot(output_x, forecasts, label='Forecast', linestyle='--')

    plt.legend()
    plt.show()

def get_data_loaders(backcast_size, forecast_size, test_size_ratio=.2, batch_size=512,
                     dataset_path='../DataCollection/aug16-2024-2yrs.parquet', dataset_col='close',
                     include_volume=False, include_velocity=False):
    data = read_processed_parquet(dataset_path)
    train_data, test_data = test_train_split(data, test_size_ratio)

    crude_dataset = None
    if include_velocity:
        if include_volume:
            crude_dataset = CrudeClosingAndVolumeVelocityDataset
        else:
            crude_dataset = CrudeClosingVelocityDataset
    elif include_volume:
        crude_dataset = CrudeClosingAndVolumeDataset
    else:
        crude_dataset = CrudeClosingPriceDataset

    dataset = crude_dataset(train_data, backcast_size, forecast_size, predict_col=dataset_col)
    testset = crude_dataset(test_data, backcast_size, forecast_size, predict_col=dataset_col)
    data_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size)
    return data_dataloader, test_dataloader


if __name__ == '__main__':
    data_loader, test_loader = get_data_loaders(72, 36, test_size_ratio=.2, batch_size=128)
