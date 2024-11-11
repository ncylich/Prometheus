import torch
from torch import nn
import math

test_prop = 0.2
forecast_size = 36
backcast_size = forecast_size * 1

# data = read_processed_parquet("/Users/noahcylich/Documents/Professional/Prometheus/DataCollection/aug16-2024-2yrs.parquet", expected_expiry_dist=3)
# _, test_data = test_train_split(data, test_prop)
# test_data = test_data['close'].to_numpy()
#
# start_idx = backcast_size
# end_idx = len(test_data) - forecast_size
# for i in range(start_idx, end_idx):
#     x = test_data[i - backcast_size:i]
#     y = test_data[i:i + forecast_size]
#     print(x, y)

def reward_function(stop_loss, multiple, start, actual, scale):
    def stop_trade(change):
        if stop_loss < 0:
            return change >= -stop_loss
        else:
            return change <= -stop_loss

    win = stop_loss * multiple
    def win_trade(change):
        if win > 0:
            return change >= win
        else:
            return change <= win

    for i in range(len(start)):
        diff = actual[i] - start[i]
        if stop_trade(diff, stop_loss):
            return -math.fabs(stop_loss) * scale
        if win_trade(diff):
            return math.fabs(win) * scale

    return (actual[-1] - start) * scale


class ForecastAction(nn.Module):
    def __init__(self):
        super(ForecastAction, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 2, 7),
            nn.GELU(),
            nn.Conv1d(2, 2, 5),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(4)
        )
        self.linear_block = nn.Sequential(
            nn.Linear(8, 8),
            nn.GELU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        x = self.conv_block(x)
        B, C, L = x.size()
        x = x.view(B, C * L)
        x = self.linear_block(x)
        return x

class AdaptiveMinPool1d:
    def __init__(self, output_size):
        self.max_pool = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x):
        return -1 * self.max_pool(-1 * x)

if __name__ == '__main__':
    model = ForecastAction()
    example_input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17]]], dtype=torch.float32)
    forecast = model(example_input)
    # model.eval()
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         forecast = model(x)
    #         print(forecast, y)
    #         break


