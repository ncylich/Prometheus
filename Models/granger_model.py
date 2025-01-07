import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.tsatools import lagmat2ds, lagmat
from statsmodels.tools.tools import add_constant

import sys
if 'google.colab' in sys.modules:
    from Prometheus.Models.granger_causality import granger_causality
else:
    from granger_causality import granger_causality


def main():
    col1, col2 = 'GC', 'VX'
    interval = 30
    prop = .1
    max_lag = 5

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)

    # 1) Load data
    df = pd.read_parquet(f'../Local_Data/{interval}min_long_term_merged_UNadjusted.parquet')
    tickers = [col for col in df.columns if col.endswith('_close')]
    df = df[tickers]
    df = df.rename(columns={col: col.split('_')[0] for col in df.columns})

    # 2) Create train/test
    size = int(len(df) * prop)
    df = df.tail(2 * size + 1)
    velocity_df = df.pct_change().dropna()
    train_vel_df = velocity_df.head(size)
    test_vel_df = velocity_df.tail(size)
    test_df = df.tail(size + 1)

    # 3) Fit Granger model
    p, f, model, best_lag = granger_causality(train_vel_df, col1, col2, max_lag=max_lag)
    # model = model.model  # The underlying statsmodels fit

    # 4) Prepare test data for predictions
    #    Keep the DataFrame around:
    test_array = test_vel_df[[col1, col2]].to_numpy()

    #    Create lagged array
    dta = lagmat2ds(test_array, best_lag, trim="both", dropex=1)
    dtajoint = add_constant(dta[:, 1:], prepend=False)

    # 5) Make predictions
    predictions = model.predict(dtajoint)

    actuals = test_df[col1].iloc[best_lag:].values
    actual_predictions = actuals[:-1] * (1 + predictions)

    residuals = actual_predictions - actuals[1:]
    naive_residuals = actuals[1:] - actuals[:-1]

    # Calculate MSE and MAE
    print(f'Naive: MSE={np.mean(naive_residuals ** 2):.4f}, MAE={np.mean(np.abs(naive_residuals)):.4f}')
    print(f'Granger: MSE={np.mean(residuals ** 2):.4f}, MAE={np.mean(np.abs(residuals)):.4f}')

if __name__ == "__main__":
    main()
