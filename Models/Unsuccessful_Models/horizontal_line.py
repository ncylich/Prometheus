import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Train.train import train_model, get_data_loaders
from Train.train import mae_and_mse_loss, plot_forecasts, interpolate_velocities
from time import sleep
import matplotlib.pyplot as plt

'''
Closing Price
    5 Min Intervals
        forecast_size = 36 -> MAE = 0.253, MSE = 0.131
        forecast_size = 72 -> MAE = 0.356, MSE = 0.249
    10 Min Intervals
        forecast_size = 36 -> MAE = 0.36, MSE = 0.256
        forecast_size = 72 -> MAE = 0.514, MSE = 0.49
Volume
    5 Min Intervals
        forecast_size = 36 -> MAE = 159.67, MSE = 75363
        forecast_size = 72 -> MAE = 167.69, MSE = 77893
'''


forecast_size = 36
test_size_ratio = 0.2
test_col = "close"  # 'close' or 'volume'

device = torch.device("cpu")

class HorizontalLine(nn.Module):
    def __init__(self):
        super(HorizontalLine, self).__init__()

    def forward(self, x):
        return torch.zeros_like(x)

if __name__ == '__main__':
    data_loader, test_loader = get_data_loaders(forecast_size, forecast_size, test_size_ratio=test_size_ratio,
                                                batch_size=2048, dataset_col=test_col,
                                                include_velocity=1)

    model = HorizontalLine().to(device)
    test_losses = torch.tensor([0, 0], dtype=torch.float32)
    plt.close('all')  # closing all previous plots
    model.eval()
    with torch.no_grad():
        for x, y, idx in test_loader:
            forecast = model(x).squeeze(0)
            # forecast = torch.zeros_like(y)  # create demo horizontal line forecast -> MAE = 0.253, MSE = 0.131
            x, forecast, y = interpolate_velocities(x, forecast, y)

            test_losses += mae_and_mse_loss(forecast, y)
            plot_forecasts(x, y, forecast)
    test_losses /= len(test_loader)
    sleep(1e-5)
    print(f'Test MAE Loss: {test_losses[0]}, MSE Loss: {test_losses[1]}')
    sleep(1e-5)