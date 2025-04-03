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
    - Short-term momentum (5D price change)
    - Long-term momentum (20D price change)
    - Mean reversion potential (Price/30D-MA)
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
    - Sentiment change (Current - 10D avg)
'''


