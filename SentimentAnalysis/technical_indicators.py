import numpy as np
import pandas as pd
try:
    from DataCollection.security_relationship_analysis import multivariate_regression, plot_2d_graph, r_squared
    from DataCollection.multivar_r2_reduction import test_multivariate_regression
except:
    pass


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


def calculate_price_momentum(prices: pd.Series, period: int = 5) -> pd.Series:
    """
    Calculate price momentum as the percentage change over a given period.
    Parameters:
        prices (pd.Series): Series of prices.
        period (int): Period for momentum calculation (default is 5).
    Returns:
        pd.Series: Percentage change over the specified period.
    """
    return prices.pct_change(periods=period)


def calculate_mean_reversion(prices: pd.Series, ma_period: int = 30) -> pd.Series:
    """
    Calculate mean reversion potential (Price/MA ratio).

    Parameters:
        prices (pd.Series): Series of prices
        ma_period (int): Period for the moving average

    Returns:
        pd.Series: Price/MA ratio
    """
    ma = prices.rolling(window=ma_period).mean()
    return prices / ma


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for a given price series.

    Parameters:
        prices (pd.Series): Series of prices
        window (int): Window size for the moving average
        num_std (float): Number of standard deviations for the bands

    Returns:
        pd.DataFrame: DataFrame with 'Upper_Band', 'Middle_Band', 'Lower_Band', and 'Width'
    """
    middle_band = prices.rolling(window=window).mean()
    std_dev = prices.rolling(window=window).std()

    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    # Calculate Bollinger Band width (Upper - Lower) / Middle
    width = (upper_band - lower_band) / middle_band

    return pd.DataFrame({
        'Upper_Band': upper_band,
        'Middle_Band': middle_band,
        'Lower_Band': lower_band,
        'Width': width
    })


def calculate_volume_ratio(volume: pd.Series, ma_period: int = 20) -> pd.Series:
    """
    Calculate volume ratio (current volume / MA volume).

    Parameters:
        volume (pd.Series): Series of trading volumes
        ma_period (int): Period for the moving average

    Returns:
        pd.Series: Volume ratio
    """
    volume_ma = volume.rolling(window=ma_period).mean()
    return volume / volume_ma


def calculate_historical_volatility(prices: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Calculate historical volatility.

    Parameters:
        prices (pd.Series): Series of prices
        window (int): Window size for volatility calculation
        annualize (bool): Whether to annualize the volatility (√252)

    Returns:
        pd.Series: Historical volatility
    """
    # Calculate daily returns
    returns = prices.pct_change().dropna()

    # Calculate rolling standard deviation of returns
    volatility = returns.rolling(window=window).std()

    # Annualize if requested (√252 for daily data)
    if annualize:
        volatility = volatility * np.sqrt(252)

    return volatility