import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy.dialects.mssql.information_schema import sequences
from statsmodels.tsa.stattools import grangercausalitytests, coint
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import contextlib
import io

# =========================
# 1. Multivariate Linear Regression
# =========================
def multivariate_regression(data, target_col):
    other_cols = [col for col in data.columns if col != target_col]

    X = data[target_col].values.reshape(-1, 1)  # Independent variable
    y = data[other_cols].values  # Dependent variable

    model = LinearRegression()
    model.fit(X, y)

    # print("Multivariate Linear Regression Results:")
    # print(f"Coefficient: {model.coef_[0]}")
    # print(f"Intercept: {model.intercept_}")
    # print(f"R^2 Score: {model.score(X, y)}")

    return model.score(X, y)

# =========================
# 2. Granger Causality Test
# =========================
def granger_causality(data, col1, col2, max_lag=5, **kwargs):
    # print("\nGranger Causality Test:")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = grangercausalitytests(data[[col1, col2]], max_lag)
    # Extract p-values and F-statistics
    p_values = [results[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag + 1)]
    f_stats = [results[lag][0]['ssr_chi2test'][0] for lag in range(1, max_lag + 1)]

    # Get the minimum p-value and corresponding lag
    min_p_value = min(p_values)
    best_lag = p_values.index(min_p_value) + 1  # Lags are 1-indexed

    # print(f"Minimum P-value: {min_p_value}")
    # print(f"Best Lag: {best_lag}")
    # print(f"Corresponding F-statistic: {f_stats[best_lag - 1]}")
    #
    # return min_p_value, best_lag, f_stats[best_lag - 1]

    return min_p_value

# =========================
# 3. Cointegration Analysis
# =========================
def cointegration_test(data, col1, col2, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        score, p_value, _ = coint(data[col1], data[col2])
    # print("\nCointegration Analysis:")
    # print(f"Cointegration Test Statistic: {score}")
    # print(f"P-Value: {p_value}")
    # if p_value < 0.05:
    #     print("The two sequences are cointegrated.")
    # else:
    #     print("The two sequences are NOT cointegrated.")
    return p_value

# =========================
# 4. Dynamic Time Warping (DTW)
# =========================
def dynamic_time_warping(data, col1, col2, **kwargs):
    sequence_a = data[col1].values
    sequence_b = data[col2].values
    distance, path = fastdtw(sequence_a, sequence_b)
    # print("\nDynamic Time Warping:")
    # print(f"DTW Distance: {distance}")
    return distance


def mat_results(data, compare_func, **kwargs):
    cols = data.columns
    results = pd.DataFrame()
    for i, col1 in enumerate(cols):
        row = []
        for j, col2 in enumerate(cols):
            if i <= j:
                row.append(compare_func(data, col1, col2, **kwargs))
            else:
                row.append(results[col2][i])
        results[col1] = row
    return results


def test():
    # Example sequences A and B (replace with your data)
    np.random.seed(42)  # For reproducibility
    A = np.cumsum(np.random.randn(100))
    B = np.cumsum(np.random.randn(100))

    data = pd.DataFrame({'A': A, 'B': B})

    multivariate_regression(data, 'A')
    granger_causality(data, 'A', 'B', max_lag=5)
    cointegration_test(data, 'A', 'B')
    dynamic_time_warping(data, 'A', 'B')

def main():
    interval = 30
    prop = .1

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)

    start = time.time()

    df = pd.read_parquet(f'../Local_Data/{interval}min_long_term_merged_UNadjusted.parquet')
    tickers = [col for col in df.columns if col.endswith('_close')]
    df = df.tail(int(len(df) * prop))
    df = df[tickers]  # remove all columns that are not tickers
    df = df.rename(columns={col: col.split('_')[0] for col in df.columns})

    corr_matrix = df.corr()
    multivar_reg = pd.DataFrame({col: [multivariate_regression(df, col)] for col in df.columns})
    granger_results = mat_results(df, granger_causality, max_lag=5)  # takes longest by far
    coint_results = mat_results(df, cointegration_test)
    dtw_results = mat_results(df, dynamic_time_warping)

    print('Correlation Coeff Mat')
    print(corr_matrix)
    print('X' * 100, '\n')

    print('Multivariate Correlation Coefficients (each col with respect to ALL of the others)')
    print(multivar_reg)
    print('X' * 100, '\n')

    print('Granger Causality Results')
    print(granger_results)
    print('X' * 100, '\n')

    print('Cointegration Results')
    print(coint_results)
    print('X' * 100, '\n')

    print('Dynamic Time Warping Results')
    print(dtw_results)
    print('X' * 100, '\n')

    print(f'Time elapsed: {time.time() - start:.2f} seconds')

if __name__ == "__main__":
    # test()
    main()