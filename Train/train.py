from DataCollection.data_processing import read_processed_parquet, test_train_split, read_parquet_nixtla
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrudeClosingPriceDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, predict_col='close'):
        self.data = data[predict_col].to_numpy()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.data) - self.backcast_size - self.forecast_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.backcast_size]
        y = self.data[idx + self.backcast_size:idx + self.backcast_size + self.forecast_size]
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)


class CrudeClosingAndVolumeDataset(Dataset):
    def __init__(self, data, backcast_size, forecast_size, predict_col='close'):
        self.price_data = data[predict_col].to_numpy()
        self.volume_data = data['volume'].to_numpy()
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
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, volume_included=False):
    model.train()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for x, y in train_loader:
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
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                forecast = model(x).squeeze(0)
                test_losses += mae_and_mse_loss(forecast, y, remove_volume=volume_included)
        test_losses /= len(test_loader)
        sleep(1e-5)
        print(f'Test MAE Loss: {test_losses[0]}, MSE Loss: {test_losses[1]}')
        sleep(1e-5)


def mae_and_mse_loss(forecast, actual, remove_volume=False):
    return torch.tensor([F.l1_loss(forecast, actual), F.mse_loss(forecast, actual)])


def get_data_loaders(backcast_size, forecast_size, test_size_ratio=.2, batch_size=512,
                     dataset_path='../DataCollection/aug16-2024-2yrs.parquet', dataset_col='close',
                     include_volume=False):
    data = read_processed_parquet(dataset_path)
    train_data, test_data = test_train_split(data, test_size_ratio)

    CrudeDataset = CrudeClosingAndVolumeDataset if include_volume else CrudeClosingPriceDataset
    dataset = CrudeDataset(train_data, backcast_size, forecast_size, predict_col=dataset_col)
    testset = CrudeDataset(test_data, backcast_size, forecast_size, predict_col=dataset_col)
    data_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size)
    return data_dataloader, test_dataloader


if __name__ == '__main__':
    data_loader, test_loader = get_data_loaders(72, 36, test_size_ratio=.2, batch_size=128)
