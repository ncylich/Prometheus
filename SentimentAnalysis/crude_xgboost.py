'''
Purpose: Crude Oil Price Prediction using XGBoost and Sentiment Analysis using Crude Bert
Large Input Features
1. Price Features
    - Crude Oil Price percent changes (1D, 5D, 10D, 30D)
    - Price ratio to moving averages (5D, 30D)
2. Momentum Indicators
    - Crude Oil MACD (fast_period=12D, slow_period=26D, signal_period=9D)
    - Crude Oil RSI (period=14D)
    - Crude Oil Stochastic (k_period=14D, d_period=3D)
3. Volatility Measures:
    - ATR (Average True Range): 14D
    - Bollinger Band width (20D, 2-std)
    - Historical volatility (10D, 20D, 30D)
4. Volume Features
    - Volume (1D, 5D, 10D, 30D)
    - Volume moving average ratios (5D/20D)
6. Crude Oil Sentiment Score
    - Sentiment score from Crude Bert (1D, 5D, 10D, 30D)
    - Avg sentiment score (5D, 30D)

Simplified Input Features
1. Price Features
    - Current Price
    - Short-term momentum (5D price change)
    - Long-term momentum (20D price change)
    - 5D moving average
    - 20D moving average
2. Technical Indicators
    - RSI (14D)
    - MACD Difference (histogram)
    - Stochastic %K (14D)
    - Bollinger Band width (20D)
3. Volume Features
    - Volume ratio (Current/20D-MA)
4. Sentiment Features
    - Daily sentiment score (1D)
    - Recent sentiment (5D avg)
    - Sentiment reversion (1D - 30D)
'''

import technical_indicators as ti

def feature_preprocessing(df, sentiment_df):
    """
    Preprocess the DataFrame by adding technical indicators and sentiment features.
    """
    # Add technical indicators
    df = ti.aggregate_to_daily(df)

    prices = df['close']
    volumes = df['volume']

    # Price features
    five_momentum = ti.calculate_price_momentum(prices, period=5)  # Series
    twenty_momentum = ti.calculate_price_momentum(prices, period=20)  # Series
    five_moving_average = prices.rolling(window=5).mean()  # Series
    twenty_moving_average = prices.rolling(window=20).mean()  # Series

    # Technical indicators
    rsi = ti.calculate_rsi(prices, period=14)  # Series
    stochastic = ti.calculate_stochastic_oscillator(prices, k_period=14, d_period=3)  # DataFrame
    macd = ti.calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)  # DataFrame
    bollinger = ti.calculate_bollinger_bands(prices, window=20, num_std=2)  # DataFrame

    # Volume features
    volume_ratio = ti.calculate_volume_ratio(volumes, ma_period=30)  # Series

    # Sentiment features
    sentiment_df = sentiment_df.set_index('date')
    sentiment = sentiment_df['avg_sentiment']
    five_avg_sent = sentiment_df['avg_sentiment'].rolling(window=5).mean()  # Series
    twenty_avg_sent = sentiment_df.rolling(window=20).mean()

    # Combine all features into the DataFrame
    df['5D_Momentum'] = five_momentum
    df['20D_Momentum'] = twenty_momentum
    df['5D_Moving_Average'] = five_moving_average
    df['20D_Moving_Average'] = twenty_moving_average

    df['RSI'] = rsi
    df['Stochastic_%K'] = stochastic['%K']
    df['Stochastic_%D'] = stochastic['%D']
    df['MACD_Line'] = macd['MACD_Line']
    df['Signal_Line'] = macd['Signal_Line']
    df['MACD_Diff'] = macd['Difference']
    df['Bollinger_Width'] = bollinger['Bollinger_Width']

    df['Volume_Ratio'] = volume_ratio

    df['Sentiment'] = sentiment
    df['5D_Avg_Sentiment'] = five_avg_sent
    df['20D_Avg_Sentiment'] = twenty_avg_sent

    # Drop rows with NaN values
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df
