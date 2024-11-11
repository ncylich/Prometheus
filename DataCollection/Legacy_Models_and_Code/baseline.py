import time

import pandas as pd
import numpy as np
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.losses.pytorch import MAE
import neuralforecast.core as NeuralForecast
from statsforecast.models import AutoARIMA
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from DataCollection.Old_Nixtla_Methods.data_processing import read_processed_parquet, print_nf_forecasts, print_np_forecasts, test_train_split
from tqdm import tqdm
import logging
import torch
import yaml
# DO NOT USE SKLEARN (messes up neuralforecast & statsforecast)
import os

# Set the environment variable to suppress the warning
os.environ['NIXTLA_ID_AS_COL'] = '1'
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

num_samples = 10
use_saved = 0

forecast_size = 36
test_set_prop = 0.2
test_sample_size = 100

input_multiple_upper_bound = 8  # 8, 10 are best for NHITs
input_size = input_multiple_upper_bound * forecast_size

random_seed = np.random.randint(37)

df_prepared = read_processed_parquet("../aug16-2024-2yrs.parquet", reset_times=True, expected_expiry_dist=3)
df_prepared.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)
df_prepared = df_prepared[['ds', 'y']]

# Convert the data to a simple integer index, necessary for models like NHITS and NBEATS
df_prepared['unique_id'] = 'stock_value'

# Results: MAE loss = 0.39, MSE loss = .326, literally a horizontal line
# Volume: Train all at once: MAE = 115, MSE = 36,500, Train for each: MAE = 115, MSE = 36,500
def arima():
    train_for_each = True

    global df_prepared

    # StatsForecast setup for AutoARIMA
    print('Training Arima...')
    arima = AutoARIMA()
    sf = StatsForecast(models=[arima], freq='5min')
    if not train_for_each:
        sf.fit(df_prepared)

    total_loss = np.array([0.0, 0.0])
    total_slope = 0
    input_data, actual, forecast = None, None, None

    np.random.seed(42)
    test_idxs = np.random.randint(0, len(df_prepared) - input_size - forecast_size, test_sample_size)
    for idx in test_idxs:
        input_data = df_prepared.iloc[idx:idx + input_size]
        if train_for_each:
            input_data = df_prepared.iloc[idx:idx + input_size]
            sf.fit(input_data)
            forecast = sf.predict(h=forecast_size)
        else:
            forecast = sf.forecast(h=forecast_size, df=input_data)
        actual = df_prepared.iloc[idx + input_size:idx + input_size + forecast_size]
        loss = test_error(forecast, 'AutoARIMA', actual)
        total_loss += loss

        slope = (actual['y'].iloc[-1] - actual['y'].iloc[0]) / forecast_size
        total_slope += slope if slope >= 0 else -slope
        # print(f'Loss: {loss}')

    print(f'Average Loss: {total_loss/test_sample_size}')
    print(f'Average Slope: {total_slope/test_sample_size}')
    print_nf_forecasts(input_data, actual, forecast, 'AutoARIMA')


def test_error(forecast, forecast_col, actual):
    diff = forecast[forecast_col].to_numpy() - actual['y'].to_numpy()
    return np.array([np.mean(np.abs(diff)), np.mean(diff ** 2)])

def avg_actual_error(forecast, actual):
    forecast.drop(columns=['ds', 'unique_id'], inplace=True)
    forecast['avg'] = forecast.mean(axis=1)
    return test_error(forecast, 'avg', actual)

def predict_test(nf, test, col_name=""):
    input_data, forecast, actual = None, None, None
    total_loss = np.array([0.0, 0.0])

    start_idx = input_multiple_upper_bound * forecast_size
    # start_idx = forecast_size
    final_idx = len(test) - forecast_size

    # Suppress PyTorch Lightning messages
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # Suppress TQDM messages
    tqdm.disable = True

    np.random.seed(42)
    test_idxs = np.random.randint(start_idx, final_idx, test_sample_size)
    # test_idxs = range(start_idx, final_idx)
    results = {}
    for i in tqdm(test_idxs):
        #input_data = test.iloc[i - input_multiple_upper_bound * forecast_size:i]
        input_data = test.iloc[i - forecast_size:i]
        forecast = nf.predict(input_data)

        # Horizontal Line
        # forecast = [input_data['y'].to_numpy()[-1]] * forecast_size
        # forecast = pd.DataFrame({col_name: forecast})

        actual = test.iloc[i:i+forecast_size]
        # calculate L1 loss
        if len(forecast.columns) > 3:
            loss = avg_actual_error(forecast, actual)
        else:
            loss = test_error(forecast, col_name, actual)
        total_loss += loss

        results[i] = list(forecast[col_name])
        print_nf_forecasts(input_data, actual, forecast, col_name)
        # print_np_forecasts(input_data, actual, forecast)
    # result_df = pd.DataFrame(results).T
    # result_df.to_csv(f'nf_results.csv')

    total_loss /= len(test_idxs)
    print(f'Average Loss: {total_loss}')
    # print_forecasts(input_data, actual, forecast, col_name)
    return total_loss


def setup_and_train_nf_model(model_class):
    model_name = model_class.__name__
    print(f'Training {model_name}...')
    model = model_class(h=forecast_size, num_samples=num_samples)
    return train_nf_model(model, model_name)

def train_nf_model(models, model_name=""):
    if not isinstance(models, list):
        models = [models]
    global df_prepared
    df_train, df_test = test_train_split(df_prepared, test_set_prop)

    if use_saved:
        nf = NeuralForecast.load('trained_NHITS_model')
        nf.save(f'trained_{model_name}_model', overwrite=True, save_dataset=True)
    else:
        nf = NeuralForecast(models=models, freq='5min')
        nf.fit(df_train)

    return predict_test(nf, df_test, model_name)

def load_model_from_yaml(yaml_path, model_class):
    with open(yaml_path, 'r') as file:
        hparams = yaml.safe_load(file)
    hparams['loss'] = MAE()
    model = model_class(**hparams)
    return model

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
    for input_multiple in [1]:
        regression_input_size = int(input_multiple * forecast_size)

        final_idx = len(df_prepared) - forecast_size - regression_input_size
        np.random.seed(42)
        test_idxs = np.random.randint(0, final_idx, test_sample_size)

        total_loss = 0.0
        for i in test_idxs:
            input_data = df_prepared.iloc[i:i + regression_input_size]

            m, b = linear_func_predictor(input_data['y'].to_numpy())
            forecast = predict_linear_regression(regression_input_size, forecast_size, m, b)
            forecast_df = pd.DataFrame({'Linear Regression': forecast})

            actual = df_prepared.iloc[i + regression_input_size:i + regression_input_size + forecast_size]
            loss = test_error(forecast_df, 'Linear Regression', actual)
            total_loss += loss
            # print(f'Loss: {loss}')
            print_np_forecasts(input_data, actual, forecast)
            time.sleep(1e-1)
        avg_loss = total_loss / len(test_idxs)
        print(f'Average Loss for input multiple {input_multiple}: {avg_loss}')
        losses[input_multiple] = avg_loss
    # losses = sorted(losses.items(), key=lambda x: x[1])
    # print(f'Best input multiple: {losses[0][0]}, Loss: {losses[0][1]}')


# best w/ input multiple 8, MAE loss 0.77
# best w/ input multiple 2, MSE loss 0.72
def linear_regression_test():
    return linear_test(linear_func_predictor=predict_linear_regression)

def horizontal_line_test():
    return linear_test(linear_func_predictor=lambda y: (0, y[-1]))


def get_scheduler(scheduler_type: str):
    if scheduler_type == 'step':
        return  {
            'lr_scheduler': torch.optim.lr_scheduler.StepLR,
            'lr_scheduler_kwargs': {
            'step_size': 10,  # Number of epochs between each learning rate decay
            'gamma': 0.1  # Multiplicative factor of learning rate decay
            }
        }
    elif scheduler_type == 'cosine':
        return {
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
            'lr_scheduler_kwargs': {
                'T_max': 100,
                'eta_min': 0.0001
            }
        }
    elif scheduler_type == 'ReduceLROnPlateau':
        raise Exception('ReduceLROnPlateau not supported')
        # return {
        #     'lr_scheduler': {
        #         'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        #         'monitor': 'val_loss',
        #     },
        #     'lr_scheduler_kwargs': {
        #         'mode': 'min',
        #         'factor': 0.1,
        #         'patience': 10,
        #         'threshold': 0.0001,
        #         'threshold_mode': 'rel',
        #         'cooldown': 0,
        #         'min_lr': 0,
        #         'eps': 1e-08,
        #         'verbose': False
        #     }
        # }
    else:
        return {
            'lr_scheduler': None,
            'lr_scheduler_kwargs': None
        }

def custom_nhits_params(scheduler_type='step', hidden_dim=512):
    # Define the parameters
    loss = MAE()

    hparams = {
        'activation': 'ReLU',
        'alias': None,
        'batch_size': 256,
        'drop_last_loader': False,
        'dropout_prob_theta': 0.0,
        'early_stop_patience_steps': -1,
        'exclude_insample_y': False,
        'h': forecast_size,
        'inference_windows_batch_size': -1,
        'input_size': forecast_size,
        'interpolation_mode': 'linear',
        'learning_rate': 0.006122658283000331,
        'loss': loss,
        'max_steps': 500.0,
        'mlp_units': [[hidden_dim] * 2] * 3,
        'n_blocks': [1, 1, 1],
        'n_freq_downsample': [24, 12, 1],
        'n_pool_kernel_size': [16, 8, 1],
        'num_lr_decays': 3,
        'num_workers_loader': 0,
        'optimizer': torch.optim.AdamW,
        'optimizer_kwargs': None,
        'pooling_mode': 'MaxPool1d',
        'random_seed': 10,
        'scaler_type': None,
        'stack_types': ['identity'] * 3,
        'start_padding_enabled': False,
        'stat_exog_list': None,
        'step_size': 36,
        'val_check_steps': 100,
        'valid_batch_size': None,
        'valid_loss': loss,
        'windows_batch_size': 256
    }
    hparams.update(get_scheduler(scheduler_type))
    if scheduler_type == 'step':
        hparams['lr_scheduler_kwargs']['step_size'] = 20
    return hparams


def custom_nbeats_params(scheduler_type='step', hidden_dim=512):
    hparams = {
        'activation': 'ReLU',
        'alias': None,
        'batch_size': 64,
        'drop_last_loader': False,
        'dropout_prob_theta': 0.0,
        'early_stop_patience_steps': -1,
        'exclude_insample_y': False,
        'futr_exog_list': None,
        'h': 36,
        'hist_exog_list': None,
        'inference_windows_batch_size': -1,
        'input_size': 72,
        'learning_rate': 0.0008120532108251931,
        'loss': MAE(),
        'max_steps': 500,
        'mlp_units': [[hidden_dim] * 2] * 3,  # Repeated blocks
        'n_blocks': [1, 1, 1],
        'n_harmonics': 2,
        'n_polynomials': 2,
        'num_lr_decays': 3,
        'num_workers_loader': 0,
        'optimizer': torch.optim.AdamW,
        'optimizer_kwargs': None,
        'random_seed': 3,
        'scaler_type': 'standard',
        'shared_weights': False,
        'stack_types': ['identity', 'trend', 'seasonality'],  # ['identity', 'trend', 'seasonality']  # trend block type doesn't matter
        'start_padding_enabled': False,
        'stat_exog_list': None,
        'step_size': 1,
        'val_check_steps': 100,
        'valid_batch_size': None,
        'valid_loss': MAE(),  # Simplified representation of the validation loss
        'windows_batch_size': 1024
    }
    hparams.update(get_scheduler(scheduler_type))
    return hparams


def train_ensemble():
    nhits1 = custom_nhits_params(hidden_dim=1024)
    nhits2 = custom_nhits_params()
    nhits2['n_blocks'] = [2, 2, 2, 2]
    nhits2['n_freq_downsample'] = [24, 12, 3, 1]
    nhits2['n_pool_kernel_size'] = [16, 8, 2, 1]


    nbeats1 = custom_nbeats_params()
    nbeats2 = custom_nbeats_params(hidden_dim=1024)
    nbeats2['stack_types'] = ['identity'] * 3
    nbeats2['input_size'] = 4 * forecast_size

    nhits1 = NHITS(**nhits1)
    nhits2 = NHITS(**nhits2)
    nbeats1 = NBEATS(**nbeats1)
    nbeats2 = NBEATS(**nbeats2)

    models = [
        nhits1,
        # nhits2,
        # nbeats1,
        # nbeats2
    ]

    train_nf_model(models, 'Ensemble')


# Uncomment the model you want to run
# arima()

# NBEATS: MAE loss = 0.23, MSE loss = .13 - Much better than ARIMA
# Volume: MAE = 84, MSE = 25000
# train_nf_model(AutoNBEATS)

# NHITS: MAE loss = 0.226, MSE loss = 0.095 - Much better than ARIMA
# Volume: MAE = 85, MSE =27,500
# train_nf_model(AutoNHITS)

# linear_regression_test()
horizontal_line_test()

# AutoFEDformer: NAN???
# AtuoAutoformer: ??
# AutoInformer: MAE = .37, MSE = .23 -> barely better than a horizontal line
# AutoLSTM: Need to try again
# AutoDilatedRNN: ??

# Horizontal: MAE = 0.22435, MSE = 0.09434

# custom_nbeats()
# train_nf_model(NHITS(**custom_nhits_params()), 'NHITS')
# train_nf_model(NBEATS(**custom_nbeats_params()), 'NBEATS')
# train_ensemble()

# models_to_try = [AutoFEDformer, AutoAutoformer, AutoInformer, AutoLSTM, AutoDilatedRNN]
# results = {}
# models_to_try = [AutoNBEATS]
# for model in models_to_try:
#     model_name = model.__name__
#     result = setup_and_train_nf_model(model)
#     results[model_name] = result
#     print(f'{model_name}: MAE loss = {result[0]}, MSE loss = {result[1]}')
# results = sorted(results.items(), key=lambda x: x[1][0])
# for model, result in results:
#     print(f'{model}: MAE loss = {result[0]}, MSE loss = {result[1]}')


