import numpy as np
import pandas as pd
from security_relationship_analysis import multivariate_regression, plot_2d_graph, r_squared
from multivar_r2_reduction import test_multivariate_regression

def aggregate_to_daily(df):
    df = df.copy()
    df['date'] = df['date'].dt.date
    aggregator = {}
    for col in df.columns:
        if 'volume' in col:
            aggregator[col] = 'sum'
        elif 'close' in col:
            aggregator[col] = 'last'
        elif 'open' in col:
            aggregator[col] = 'first'
    df = df.groupby('date').agg(aggregator)

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Parameters:
        prices (pd.Series): Series of prices.
        period (int): The period to use for RSI calculation (default is 14).

    Returns:
        pd.Series: RSI values.
    """
    # Compute the difference in prices
    delta = prices.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate the rolling means of gains and losses
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Compute the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_rsi_for_tickers(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate the RSI for each ticker in the DataFrame.
    Assumes that ticker close columns are named with the pattern '{ticker}_close'.

    Parameters:
        df (pd.DataFrame): DataFrame with a datetime index or date column and ticker columns.
        period (int): The period to use for the RSI calculation (default is 14).

    Returns:
        pd.DataFrame: DataFrame with additional RSI columns for each ticker.
    """
    # Create a copy of DataFrame to add RSI columns
    df = df.copy()

    # Determine tickers by identifying columns ending with '_close'
    ticker_list = [col.split('_')[0] for col in df.columns if col.endswith('_close')]

    # Calculate the RSI for each ticker and add as a new column
    for ticker in ticker_list:
        close_col = f'{ticker}_close'
        rsi_col = f'{ticker}_rsi'
        df[rsi_col] = calculate_rsi(df[close_col], period)

    return df


def calculate_stochastic_oscillator(prices: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate the stochastic oscillator (%K and %D) for a given price series.

    Parameters:
        prices (pd.Series): Series of closing prices.
        k_period (int): Period for calculating the %K (default is 14).
        d_period (int): Period for calculating the %D (default is 3).

    Returns:
        pd.DataFrame: DataFrame with columns '%K' and '%D'
    """
    # Calculate rolling minimum and maximum over the k_period
    low_min = prices.rolling(window=k_period, min_periods=k_period).min()
    high_max = prices.rolling(window=k_period, min_periods=k_period).max()

    # Calculate %K
    percent_k = (prices - low_min) / (high_max - low_min) * 100

    # Calculate %D as the rolling mean of %K
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({'%K': percent_k, '%D': percent_d})


def calculate_stochastic_for_tickers(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate the stochastic oscillator for each ticker in the DataFrame.
    Assumes that ticker close columns are named with the pattern '{ticker}_close'.

    Parameters:
        df (pd.DataFrame): DataFrame with a date column and ticker columns.
        k_period (int): Period for calculating %K (default is 14).
        d_period (int): Period for calculating %D (default is 3).

    Returns:
        pd.DataFrame: DataFrame with additional columns for each ticker's stochastic oscillator:
                      '{ticker}_stoch_%K' and '{ticker}_stoch_%D'.
    """
    # Create a copy so as not to modify the original DataFrame
    df = df.copy()

    # Determine tickers based on columns ending with '_close'
    ticker_list = [col.split('_')[0] for col in df.columns if col.endswith('_close')]

    # Calculate stochastic oscillator for each ticker and add the columns to df
    for ticker in ticker_list:
        close_col = f'{ticker}_close'
        stoch_data = calculate_stochastic_oscillator(df[close_col], k_period=k_period, d_period=d_period)
        df[f'{ticker}_stoch_%K'] = stoch_data['%K']
        df[f'{ticker}_stoch_%D'] = stoch_data['%D']

    return df


def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate the MACD (Moving Average Convergence Divergence) for a given price series.

    Parameters:
        prices (pd.Series): Series of prices.
        fast_period (int): Period for the fast EMA (default is 12).
        slow_period (int): Period for the slow EMA (default is 26).
        signal_period (int): Period for the signal line EMA (default is 9).

    Returns:
        pd.DataFrame: DataFrame with columns 'MACD_Line', 'Signal_Line', and 'Histogram'.
    """
    # Calculate the Exponential Moving Averages (EMAs)
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line as the difference between the EMAs
    macd_line = ema_fast - ema_slow

    # Calculate the Signal line as the EMA of the MACD line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate the Histogram as the difference between the MACD line and the Signal line
    macd_diff = macd_line - signal_line

    return pd.DataFrame({
        'MACD_Line': macd_line,
        'Signal_Line': signal_line,
        'Difference': macd_diff
    })


def calculate_macd_for_tickers(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                               signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate the MACD for each ticker in the DataFrame.
    Assumes that ticker close columns are named with the pattern '{ticker}_close'.

    Parameters:
        df (pd.DataFrame): DataFrame with a date column and ticker columns.
        fast_period (int): Fast EMA period (default is 12).
        slow_period (int): Slow EMA period (default is 26).
        signal_period (int): Signal EMA period (default is 9).

    Returns:
        pd.DataFrame: DataFrame with additional MACD columns for each ticker:
                      '{ticker}_macd' for the MACD line,
                      '{ticker}_signal' for the Signal line,
                      '{ticker}_histogram' for the Histogram.
    """
    # Create a copy so as not to modify the original DataFrame
    df = df.copy()

    # Determine tickers based on columns ending with '_close'
    ticker_list = [col.split('_')[0] for col in df.columns if col.endswith('_close')]

    # Calculate MACD for each ticker and add the columns to df
    for ticker in ticker_list:
        close_col = f'{ticker}_close'
        macd_data = calculate_macd(df[close_col], fast_period=fast_period, slow_period=slow_period,
                                   signal_period=signal_period)
        df[f'{ticker}_macd'] = macd_data['MACD_Line']
        df[f'{ticker}_signal'] = macd_data['Signal_Line']
        df[f'{ticker}_difference'] = macd_data['Difference']

    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns - {'date'}
    tickers = [col.split('_')[0] for col in cols if col.endswith('_close')]
    col_types = ['_'.join(col.split('_')[1:]) for col in cols]

    type_order = []
    found = set()
    for col_type in col_types:
        if col_type not in found:
            type_order.append(col_type)
            found.add(col_type)

    ordered_cols = ['date']
    for ticker in tickers:
        for col_type in type_order:
            ordered_cols.append(f'{ticker}_{col_type}')

    return df[ordered_cols]


def linear_predict(data, target_col, beta, degree=1):
    target_col_suffix = '_'.join(target_col.split('_')[1:])
    other_cols = [col for col in data.columns if col != target_col and col.endswith(target_col_suffix)]

    start_X = data[other_cols].values
    X = start_X  # Independent variable

    for i in range(2, degree + 1):
        X = np.concatenate((X, start_X ** i), axis=1)

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    pred = X.dot(beta)
    return pred


def linear_ticker_regression(df, ticker, train_prop=0.8, degree=1):
    org_df = df.copy()
    cols = [col for col in df.columns if col.endswith('_close')]
    for col in cols:
        df[col] = df[col].pct_change()
    df = df[cols].dropna()

    test_df = df.tail(int(len(df) * train_prop))
    train_df = df.head(len(df) - int(len(df) * train_prop))

    col = ticker + "_close"

    score, model = multivariate_regression(train_df, col, degree=degree)
    print(f"{col} train: {score}")

    test_score = test_multivariate_regression(model, test_df, col, degree=degree)
    print(f"{col} test: {test_score}")

    pred = linear_predict(df, col, model, degree=degree)
    org_df[f'{col}_pred'] = pred
    return org_df


def main():
    df = pd.read_parquet(f'../Local_Data/focused_futures_30min/interpolated_all_long_term_combo.parquet')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')

    df = aggregate_to_daily(df)

    df = calculate_rsi_for_tickers(df, period=14)
    df = calculate_stochastic_for_tickers(df, k_period=14, d_period=3)
    df = calculate_macd_for_tickers(df, fast_period=12, slow_period=26, signal_period=9)

    df = df.dropna()

    # cl_cols = df.columns[df.columns.str.startswith('CL')].tolist()
    # # print(cl_cols)
    # df = df[cl_cols]
    # df.to_csv("../Local_Data/technical_indicators.csv", index=True)

    df = linear_ticker_regression(df, 'CL')

    return df


if __name__ == '__main__':
    main()
