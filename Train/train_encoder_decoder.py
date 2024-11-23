from time import sleep
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiStockClosingAndVolumeDataset(Dataset):
    SCALE_VEL = 10

    def __init__(self, data, backcast_size, forecast_size, predict_col='close', tickers=None):
        # columns in data csv: ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        if tickers is None:
            tickers = set([col.split('_')[0] for col in list(data.columns) if '_' in col])

        data['date'] = pd.to_datetime(data['date'], errors='coerce', utc=True)

        self.tickers = tickers
        self.velocities = {ticker: MultiStockClosingAndVolumeDataset.SCALE_VEL *
            self.calculate_velocity(torch.from_numpy(data[ticker + '_' + predict_col].to_numpy()).float()) for ticker in tickers}
        self.volumes = {ticker: torch.from_numpy(data[ticker + '_volume'].to_numpy()).float() / 1e3 for ticker in tickers}
        self.price_data = torch.from_numpy(data['CL_' + predict_col].to_numpy()[1:]).float()
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
            y_price = price_seq[self.backcast_size:]
            y_volume = volume_seq[self.backcast_size:]

            x_prices.append(x_price)
            x_volumes.append(x_volume)
            y_prices.append(y_price)
            y_volumes.append(y_volume)

        x = torch.stack(x_prices + x_volumes)  # Shape: [2, num_tickers, backcast_size]
        y = torch.stack(y_prices + y_volumes)  # Shape: [2, num_tickers, backcast_size + forecast_size]

        time = self.times[idx + self.backcast_size]

        return x.to(device), y.to(device), time.to(device), CL_price_seq.to(device)

    def calculate_velocity(self, data):
        return data[1:] - data[:-1]


def plot_forecast_vs_actual(forecast, actual):
    plt.figure(figsize=(9, 6))
    plt.plot(forecast, label='Forecast')
    plt.plot(actual, label='Actual')
    # plt.plot(gt_seq, label='Ground Truth')
    plt.legend()
    plt.show()


def test_train_split(df, test_size_ratio):
    test_len = int(len(df) * test_size_ratio)
    return df.head(len(df) - test_len).copy(), df.tail(test_len).copy()


def train_model_together(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    model.to(device)

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for (i, (x, y, t, gt_seq)) in enumerate(train_loader):
            x, y, t = x.to(device), y.to(device), t.to(device)
            optimizer.zero_grad()
            forecast = model(x, t)
            loss = criterion(forecast, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}')
        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')

        num_example_plots = 10
        if num_example_plots > len(test_loader):
            plot_idxs = set(range(len(test_loader)))
        else:
            plot_idxs = set(random.sample(range(len(test_loader)), num_example_plots))

        test_losses = torch.tensor([0,0], dtype=torch.float32)
        model.eval()
        total_correct_ups = 0
        total_correct_downs = 0
        total_correct_overall = 0
        with torch.no_grad():
            for i, (x, y, t, gt_seq) in enumerate(test_loader):
                x, t = x.to(device), t.to(device)
                output = model(x, t)
                # try:
                #     forecast = model.dct_backward(output) # [batch_size, V, seq_len]
                # except AttributeError:
                #     forecast = output
                forecast = model.post_process(output)  # generalizes try-catch above

                # scaling back to original values
                y /= MultiStockClosingAndVolumeDataset.SCALE_VEL
                output /= MultiStockClosingAndVolumeDataset.SCALE_VEL

                # halving values to only consider price data (if applicable)
                # print(y.shape)
                # plot_forecast_vs_actual(forecast[0], y[0])
                forecast = forecast[:, 0].squeeze(1)
                y = y[:, 0].squeeze(1)
                # plot_forecast_vs_actual(output[0][0].cpu(), y[0].cpu())

                torch.cumsum(forecast, dim=-1, out=forecast)
                torch.cumsum(y, dim=-1, out=y)

                # Squeezing to final point of input
                forecast += gt_seq[:, x.size()[-1] - 1].unsqueeze(1) - forecast[:, x.size()[-1] - 1].unsqueeze(1)
                y += gt_seq[:, x.size()[-1] - 1].unsqueeze(1) - y[:, x.size()[-1] - 1].unsqueeze(1)

                # forecast += y[:, x.size()[-1]].unsqueeze(1) - forecast[:, x.size()[-1]].unsqueeze(1)
                if i in plot_idxs:
                    plot_forecast_vs_actual(forecast[0].cpu(), y[0].cpu())

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
        scheduler.step(test_losses[1])  # only use MSE loss for scheduler

        sleep(1e-5)
        print(f'Test MAE Loss: {test_losses[0]}, MSE Loss: {test_losses[1]}')
        correct_ups = total_correct_ups/len(test_loader)
        correct_downs = total_correct_downs/len(test_loader)
        correct_overall = total_correct_overall/len(test_loader)
        f1 = 2 * correct_ups * correct_downs / (correct_ups + correct_downs)
        print(f'Correct Ups: {correct_ups}, Correct Downs: {correct_downs}')
        print(f'Correct Overall: {correct_overall}, F1 Score: {f1}')

        sleep(1e-5)


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    model.to(device)

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for i, (x, y, t, gt_seq) in enumerate(train_loader):
            x, y, t = x.to(device), y.to(device), t.to(device)
            optimizer.zero_grad()
            forecast = model(x, t)
            y_reshape = y[:, 0].unsqueeze(2)
            assert y_reshape.shape == forecast.shape
            loss = criterion(forecast, y_reshape)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}')
        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')

        num_example_plots = 10
        if num_example_plots > len(test_loader):
            plot_idxs = set(range(len(test_loader)))
        else:
            plot_idxs = set(random.sample(range(len(test_loader)), num_example_plots))

        test_losses = torch.tensor([0,0], dtype=torch.float32)
        model.eval()
        total_correct_ups = 0
        total_correct_downs = 0
        total_correct_overall = 0
        with torch.no_grad():
            for i, (x, y, t, gt_seq) in enumerate(test_loader):
                x, t = x.to(device), t.to(device)
                output = model(x, t)

                # scaling back to original values
                y /= MultiStockClosingAndVolumeDataset.SCALE_VEL
                output /= MultiStockClosingAndVolumeDataset.SCALE_VEL

                # halving values to only consider price data (if applicable)
                # print(y.shape)
                # plot_forecast_vs_actual(forecast[0], y[0])
                forecast = output.squeeze(1)
                y = y[:, 0].squeeze(1)
                # plot_forecast_vs_actual(output[0][0].cpu(), y[0].cpu())

                torch.cumsum(forecast, dim=-1, out=forecast)
                torch.cumsum(y, dim=-1, out=y)

                # Squeezing to final point of input
                forecast = forecast.squeeze(-1) + gt_seq[:, x.size()[-1] - 1].unsqueeze(1)
                y += gt_seq[:, x.size()[-1] - 1].unsqueeze(1)

                if i in plot_idxs:
                    plot_forecast_vs_actual(forecast[0].cpu(), y[0].cpu())

                # calculate percentage of correct ups, correct downs, and correct overall
                sign_truth = torch.sign(y[:, -1] - y[:, 0])
                sign_forecast = torch.sign(forecast[:, -1] - forecast[:, 0])
                correct_ups = torch.sum((sign_truth + sign_forecast) == 2) / torch.sum(sign_truth == 1)
                correct_downs = torch.sum((sign_truth + sign_forecast) == -2) / torch.sum(sign_truth == -1)
                correct_overall = torch.sum(sign_truth == sign_forecast) / len(sign_truth)
                total_correct_ups += correct_ups
                total_correct_downs += correct_downs
                total_correct_overall += correct_overall

                test_losses += mae_and_mse_loss(forecast, y)
        test_losses /= len(test_loader)
        scheduler.step(test_losses[1])  # only use MSE loss for scheduler

        sleep(1e-5)
        print(f'Test MAE Loss: {test_losses[0]}, MSE Loss: {test_losses[1]}')
        correct_ups = total_correct_ups/len(test_loader)
        correct_downs = total_correct_downs/len(test_loader)
        correct_overall = total_correct_overall/len(test_loader)
        f1 = 2 * correct_ups * correct_downs / (correct_ups + correct_downs)
        print(f'Correct Ups: {correct_ups}, Correct Downs: {correct_downs}')
        print(f'Correct Overall: {correct_overall}, F1 Score: {f1}')

        sleep(1e-5)


def mae_and_mse_loss(forecast, actual):
    return torch.tensor([F.l1_loss(forecast, actual), F.mse_loss(forecast, actual)])


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

    CrudeDataset = MultiStockClosingAndVolumeDataset
    train_dataset = CrudeDataset(train_data, backcast_size, forecast_size, predict_col=dataset_col)
    test_dataset = CrudeDataset(test_data, backcast_size, forecast_size, predict_col=dataset_col)

    data_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
    return data_dataloader, test_dataloader


if __name__ == '__main__':
    data_loader, test_loader = get_long_term_data_loaders(72, 36, test_size_ratio=.2, batch_size=128)
