import time
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, coint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import contextlib
import io


def granger_causality(data, col1, col2, max_lag=5, **kwargs):
    # print("\nGranger Causality Test:")
    col1_values = data[col1].values
    col2_values = data[col2].values
    if col1 == col2:
        col2 = col2 + '2'
    df = pd.DataFrame({col1: col1_values, col2: col2_values})

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        col_df = df[[col1, col2]]
        results = grangercausalitytests(col_df, max_lag)

    # Extract p-values and F-statistics
    p_values = [results[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag + 1)]
    f_stats = [results[lag][0]['ssr_chi2test'][0] for lag in range(1, max_lag + 1)]

    # Get the minimum p-value and corresponding lag
    min_p_value = min(p_values)
    best_lag = p_values.index(min_p_value) + 1  # Lags are 1-indexed
    best_f_stat = f_stats[best_lag - 1]
    best_model = results[best_lag][1][1]

    return min_p_value, best_f_stat, best_model, best_lag

def mat_results(data, compare_func, **kwargs):
    cols = data.columns
    results1 = pd.DataFrame()
    results2 = pd.DataFrame()
    results3 = pd.DataFrame()
    for i, col1 in enumerate(cols):
        row1 = []
        row2 = []
        row3 = []
        for j, col2 in enumerate(cols):
            p, f, model, _ = compare_func(data, col1, col2, **kwargs)
            row1.append(p)
            row2.append(f)
            row3.append(model)
        results1[col1] = row1
        results2[col1] = row2
        results3[col1] = row3
    return results1, results2, results3

def plot_heat_map(matrix, title='Heat Map'):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Height')
    plt.title(title)
    plt.xlabel('Tickers')
    plt.ylabel('Tickers')

    assert matrix.shape[0] == matrix.shape[1]
    plt.xticks(ticks=np.arange(matrix.shape[0]), labels=matrix.columns, rotation=0, ha='center')
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=matrix.columns, rotation=0, ha='right')

    # moving x-axis labels to top
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')

    plt.tight_layout()
    plt.show()

def main():
    interval = 30
    prop = .1
    max_lag = 10

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)

    start = time.time()

    df = pd.read_parquet(f'../Local_Data/{interval}min_long_term_merged_UNadjusted.parquet')
    tickers = [col for col in df.columns if col.endswith('_close')]
    df = df.tail(int(len(df) * prop))
    df = df[tickers]  # remove all columns that are not tickers
    df = df.rename(columns={col: col.split('_')[0] for col in df.columns})

    stds = []
    for col in df.columns:
        df[col] = df[col].pct_change().fillna(0)
        stds.append(df[col].std())
    print(stds)
    print(f'Avg Std of Multiplicative Velocities: {sum(stds)/len(stds)}')

    p_results, f_results, models = mat_results(df, granger_causality, max_lag=max_lag)
    plot_heat_map(p_results, title='Granger Causality P-Value Heat Map')
    plot_heat_map(f_results, title='Granger Causality F-Statistic Heat Map')

    # finding i, j of min p-value and max f-statistic
    min_p = np.min(p_results.values)
    max_f = np.max(f_results.values)
    i, j = np.where(p_results.values == min_p)
    k, l = np.where(f_results.values == max_f)

    print(f"Minimum P-value: {min_p: .8f}: {p_results.columns[j[0]]} -> {p_results.columns[i[0]]}")
    print(f"Maximum F-statistic: {max_f: .2f}: {f_results.columns[l[0]]} -> {f_results.columns[k[0]]}")

    end = time.time()
    print(f"Time Elapsed: {end - start: .2f} seconds")

if __name__ == '__main__':
    main()
