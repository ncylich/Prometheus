import pandas as pd
import numpy as np
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.auto import AutoNBEATS, AutoNHITS
from statsforecast.models import AutoARIMA
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from neuralforecast.losses.pytorch import DistributionLoss
from data_processing import read_parquet_nixtla, print_forecasts, test_train_split
# DO NOT USE SKLEARN (messes up neuralforecast & statsforecast)
import os

# Set the environment variable to suppress the warning
os.environ['NIXTLA_ID_AS_COL'] = '1'

forecast_size = 36
test_set_prop = 0.2
test_sample_size = 100

input_multiple_upper_bound = 8
input_size = input_multiple_upper_bound * forecast_size

random_seed = np.random.randint(0, int(1e6))

df_prepared = read_parquet_nixtla("aug16-2024-2yrs.parquet", smush_times=True, expected_expiry_dist=3, y_var='close')

# Convert the data to a simple integer index, necessary for models like NHITS and NBEATS
df_prepared['unique_id'] = 'stock_value'


# Results: MAE loss = 0.39, MSE loss = .326, literally a horizontal line
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
    total_slope = 0
    input_data, actual, forecast = None, None, None
    actual_input_size = input_size if train_for_each else forecast_size

    np.random.seed(42)
    test_idxs = np.random.randint(0, len(df_prepared) - actual_input_size - forecast_size, test_sample_size)
    for idx in test_idxs:
        input_data = df_prepared.iloc[idx:idx + actual_input_size]
        if train_for_each:
            input_data = df_prepared.iloc[idx:idx + actual_input_size]
            sf.fit(input_data)
            forecast = sf.predict(h=forecast_size)
        else:
            forecast = sf.forecast(h=forecast_size, df=input_data)
        actual = df_prepared.iloc[idx + actual_input_size:idx + actual_input_size + forecast_size]
        loss = test_error(forecast, 'AutoARIMA', actual)
        total_loss += loss

        slope = (actual['y'].iloc[-1] - actual['y'].iloc[0]) / forecast_size
        total_slope += slope if slope >= 0 else -slope
        # print(f'Loss: {loss}')

    print(f'Average Loss: {total_loss/test_sample_size}')
    print(f'Average Slope: {total_slope/test_sample_size}')
    print_forecasts(input_data, actual, forecast, 'AutoARIMA')


def test_error(forecast, forecast_col, actual):
    diff = forecast[forecast_col].to_numpy() - actual['y'].to_numpy()
    # return np.mean(np.abs(diff))  # L1, MAE
    return np.mean(diff ** 2)  # L2, MSE


def predict_test(nf, test, col_name):
    input_data, forecast, actual = None, None, None
    total_loss = 0
    count = 0

    start_idx = input_multiple_upper_bound * forecast_size
    final_idx = len(test) - forecast_size

    np.random.seed(42)
    test_idxs = np.random.randint(start_idx, final_idx, test_sample_size)
    for i in test_idxs:
        input_data = test.iloc[i - input_multiple_upper_bound * forecast_size:i]
        forecast = nf.predict(input_data)
        actual = test.iloc[i:i+forecast_size]
        # calculate L1 loss
        loss = test_error(forecast, col_name, actual)
        total_loss += loss
        count += 1
        # print(f'Loss: {loss}')
        # print_forecasts(input_data, actual, forecast, col_name)
    print(f'Average Loss: {total_loss/count}')
    print_forecasts(input_data, actual, forecast, col_name)


# MAE loss = 0.23, MSE loss = .13 - Much better than ARIMA
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
    df_train, df_test = test_train_split(df_prepared, test_set_prop)
    nf.fit(df_train, val_size=forecast_size)
    predict_test(nf, df_test, 'AutoNBEATS')

# MAE loss = 0.24, MSE loss = 0.13 - Much better than ARIMA
def nhits():
    # NeuralForecast setup for NHITS
    print('Training NHITS...')
    nhits_model = AutoNHITS(h=forecast_size)
    # nhits_model = NHITS(h=forecast_size, input_size=input_multiple*forecast_size, n_blocks=3, mlp_units=512)
    nf = NeuralForecast(models=[nhits_model], freq='5min')  # 'T' for minute-level data
    global df_prepared
    df_train, df_test = test_train_split(df_prepared, test_set_prop)
    nf.fit(df_train)
    predict_test(nf, df_test, 'AutoNHITS')


def linear_regression(y):
    x = np.arange(len(y))
    n = len(y)

    # Calculate the sums needed for the slope (m) and intercept (b)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)

    # Calculate slope (m) and intercept (b)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    return m, b


def predict_linear_regression(start_x, length, m, b):
    return m * np.arange(start_x, start_x + length) + b


def linear_test(linear_func_predictor):
    losses = {}
    for input_multiple in [.25, .5, 1, 2, 4, 8]:
        regression_input_size = int(input_multiple * forecast_size)

        final_idx = len(df_prepared) - forecast_size - regression_input_size
        np.random.seed(42)
        test_idxs = np.random.randint(0, final_idx, 10*test_sample_size)

        total_loss = 0.0
        for i in test_idxs:
            input_data = df_prepared.iloc[i:i + regression_input_size]

            m, b = linear_func_predictor(input_data['y'].to_numpy())
            forecast = predict_linear_regression(regression_input_size, forecast_size, m, b)
            forecast_df = pd.DataFrame({'Linear Regression': forecast})

            actual = df_prepared.iloc[i + regression_input_size:i + regression_input_size + forecast_size]
            loss = test_error(forecast_df, 'Linear Regression', actual)
            total_loss += loss
            print(f'Loss: {loss}')
            # print_forecasts(input_data, actual, forecast, 'Linear Regression')
        avg_loss = total_loss / len(test_idxs)
        print(f'Average Loss for input multiple {input_multiple}: {avg_loss}')
        losses[input_multiple] = avg_loss
    losses = sorted(losses.items(), key=lambda x: x[1])
    print(f'Best input multiple: {losses[0][0]}, Loss: {losses[0][1]}')


# best w/ input multiple 8, MAE loss 0.77
# best w/ input multiple 2, MSE loss 0.72
def linear_regression_test():
    return linear_test(linear_func_predictor=predict_linear_regression)


# NULL HYPOTHESIS TEST: If no trend is present, the best predictor is a horizontal line
# MAE loss = 0.365, MSE loss = 0.31
def horizontal_line_test():
    def horizontal_line_predictor(y):
        return 0, y[-1]
    return linear_test(linear_func_predictor=horizontal_line_predictor)


# Uncomment the model you want to run
# arima()
# nbeats()
nhits()
# linear_regression_test()
# horizontal_line_test()
