import pandas as pd
import numpy as np
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.auto import AutoNBEATS, AutoNHITS
from statsforecast.models import AutoARIMA
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from neuralforecast.losses.pytorch import DistributionLoss
from data_processing import read_parquet, print_forecasts, test_train_split
# DO NOT USE SKLEARN (messes up neuralforecast & statsforecast)
import os

# Set the environment variable to suppress the warning
os.environ['NIXTLA_ID_AS_COL'] = '1'

forecast_size = 36
test_size = 0.2
input_multiple_upper_bound = 8
input_size = input_multiple_upper_bound * forecast_size

random_seed = np.random.randint(0, int(1e6))

df_prepared = read_parquet("aug16-2024-2yrs.parquet", True)

# Convert the data to a simple integer index, necessary for models like NHITS and NBEATS
df_prepared['unique_id'] = 'stock_value'


# avg loss 0.47 (train_for_each = True) - best so far
def arima():
    train_for_each = False

    global df_prepared

    # StatsForecast setup for AutoARIMA
    print('Training Arima...')
    arima = AutoARIMA()
    sf = StatsForecast(models=[arima], freq='5min')
    if not train_for_each:
        sf.fit(df_prepared)

    total_loss = 0
    input_data, actual, forecast = None, None, None
    actual_input_size = input_size if train_for_each else forecast_size

    np.random.seed(42)
    test_idxs = np.random.randint(0, len(df_prepared) - actual_input_size - forecast_size, 30)
    for idx in test_idxs:
        input_data = df_prepared.iloc[idx:idx + actual_input_size]
        if train_for_each:
            input_data = df_prepared.iloc[idx:idx + actual_input_size]
            sf.fit(input_data)
            forecast = sf.predict(h=forecast_size)
        else:
            forecast = sf.forecast(h=forecast_size, df=input_data)
        actual = df_prepared.iloc[idx + actual_input_size:idx + actual_input_size + forecast_size]
        loss = mean_absolute_error(forecast, 'AutoARIMA', actual)
        total_loss += loss
        # print(f'Loss: {loss}')

    print(f'Average Loss: {total_loss/30}')
    print_forecasts(input_data, actual, forecast, 'AutoARIMA')


def mean_absolute_error(forecast, forecast_col, actual):
    return np.mean(np.abs(forecast[forecast_col].to_numpy() - actual['y'].to_numpy()))


def predict_test(nf, test, col_name):
    input_data, forecast, actual = None, None, None
    total_loss = 0
    count = 0

    start_idx = input_multiple_upper_bound * forecast_size
    final_idx = len(test) - forecast_size

    np.random.seed(42)
    test_idxs = np.random.randint(start_idx, final_idx, 30)
    for i in range(start_idx, final_idx):
        if i not in test_idxs:
            continue
        input_data = test.iloc[i - input_multiple_upper_bound * forecast_size:i]
        forecast = nf.predict(input_data)
        actual = test.iloc[i:i+forecast_size]
        # calculate L1 loss
        loss = mean_absolute_error(forecast, col_name, actual)
        total_loss += loss
        count += 1
        # print(f'Loss: {loss}')
    print(f'Average Loss: {total_loss/count}')
    print_forecasts(input_data, actual, forecast, col_name)


# avg loss 0.584 - barely better than regression
def nbeats():
    # NeuralForecast setup for NBEATS
    print('Training NBEATS...')
    nbeats_model = AutoNBEATS(h=forecast_size)

    # nbeats = NBEATS(h=forecast_size,
    #             input_size = input_multiple*forecast_size,
    #             n_polynomials = 5,
    #             stack_types = ['identity', 'trend', 'seasonality'],
    #             n_harmonics = 3,
    #             n_blocks = [4,4,4],
    #             # mlp_units = [[512, 2048], [512, 2048], [512, 2048]],
    #             # max_steps = 100,
    #             val_check_steps = 10,
    #             early_stop_patience_steps = 2,
    #             )

    nf = NeuralForecast(models=[nbeats_model], freq='5min')
    global df_prepared
    df_train, df_test = test_train_split(df_prepared, test_size)
    nf.fit(df_train, val_size=forecast_size)
    predict_test(nf, df_test, 'AutoNBEATS')

# avg loss 0.77 - worse than regression
def nhits():
    # NeuralForecast setup for NHITS
    print('Training NHITS...')
    nhits_model = AutoNHITS(h=forecast_size)
    # nhits_model = NHITS(h=forecast_size, input_size=input_multiple*forecast_size, n_blocks=3, mlp_units=512)
    nf = NeuralForecast(models=[nhits_model], freq='5min')  # 'T' for minute-level data
    global df_prepared
    df_train, df_test = test_train_split(df_prepared, test_size)
    nf.fit(df_train)
    predict_test(nf, df_test, 'AutoNHITS')

# Uncomment the model you want to run
arima()
# nbeats()
# nhits()
