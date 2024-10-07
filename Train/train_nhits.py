from DataCollection.data_processing import read_processed_parquet, test_train_split, read_parquet_nixtla
from Models.NBeats import BlockType
from Models.NHits import Nhits
import numpy as np
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


lr = 1e-3
batch_size = 1024
epochs = 50
init_weight_magnitude = 1e-3

forecast_size = 36
backcast_size = forecast_size * 2

stack_pools = [16, 8, 4, 2]  # [2,4,8,16]
stack_mlp_freq_downsamples = [24, 12, 4, 1]
hidden_dim = 256
n_blocks = 6
n_layers = 3

test_size_ratio = .2
test_sample_size = 100

test_col = 'close'  # 'y'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrudeDataset(Dataset):
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


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        model.train()
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

        test_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                forecast = model(x).squeeze(0)
                test_loss += test_error(forecast, y)
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss}')


def test_error(forecast, actual):
    diff = forecast.detach().numpy() - actual.detach().numpy()
    # return np.mean(np.abs(diff))  # L1, MAE
    return np.mean(diff ** 2)  # L2, MSE


if __name__ == '__main__':
    data = read_processed_parquet('../DataCollection/aug16-2024-2yrs.parquet')
    # data = read_parquet_nixtla('../DataCollection/aug16-2024-2yrs.parquet', smush_times=True, expected_expiry_dist=3)
    train_data, test_data = test_train_split(data, test_size_ratio)

    dataset = CrudeDataset(train_data, backcast_size, forecast_size, predict_col=test_col)
    testset = CrudeDataset(test_data, backcast_size, forecast_size, predict_col=test_col)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size)

    model = Nhits(backcast_size, forecast_size, n_layers=n_layers, stack_pools=stack_pools,
                  stack_mlp_freq_downsamples=stack_mlp_freq_downsamples, n_blocks=n_blocks).to(device)

    def initialize_weights(module):
        if isinstance(module, nn.Linear):
            # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            torch.nn.init.uniform_(module.weight, a=-init_weight_magnitude, b=init_weight_magnitude)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    #model.apply(initialize_weights)

    criterion = torch.nn.L1Loss()
    # criterion = MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1)
    # step down LR after so many steps

    train_model(model, data_loader, test_loader, criterion, optimizer, scheduler, epochs)

    # predict_test(model, test_data)
