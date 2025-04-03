import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    df = df[df['date'].dt.year == 2024].copy()  # Only taking 2024 data
    df = df.reset_index()

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
    sentiment = sentiment_df['avg_sentiment']
    five_avg_sent = sentiment_df['avg_sentiment'].rolling(window=5).mean()  # Series
    twenty_avg_sent = sentiment_df['avg_sentiment'].rolling(window=20).mean()

    # Combine all features into the DataFrame
    df['future_price_movement'] = df['close'].pct_change().shift(-1)

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
    df['Bollinger_Width'] = bollinger['Width']
    df['Bollinger_Upper'] = bollinger['Upper_Band']
    df['Bollinger_Middle'] = bollinger['Middle_Band']
    df['Bollinger_Lower'] = bollinger['Lower_Band']

    df['Volume_Ratio'] = volume_ratio

    df['Sentiment'] = sentiment
    df['5D_Avg_Sentiment'] = five_avg_sent
    df['20D_Avg_Sentiment'] = twenty_avg_sent

    # Drop rows with NaN values
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df


def create_and_train_model(df):
    """
    Create, train, and evaluate an XGBoost model for predicting future oil prices.

    Parameters:
        df (pd.DataFrame): Processed dataframe with features and target

    Returns:
        tuple: (trained model, evaluation metrics, feature importance)
    """
    # Ensure date is not used as a feature if present
    features_df = df.copy()
    if 'date' in features_df.columns:
        features_df = features_df.set_index('date')

    # Separate features and target
    X = features_df.drop('future_price_movement', axis=1)
    y = features_df['future_price_movement']

    # Split data chronologically (last 20% for testing)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")

    # Initialize and train XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse',
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Calculate direction prediction accuracy
    if 'close' in X_test.columns:
        actual_direction = np.sign(y_test - X_test['close'])
        predicted_direction = np.sign(y_pred - X_test['close'].values)
        metrics['direction_accuracy'] = np.mean(actual_direction == predicted_direction)

    # Print results
    print("\nModel Performance:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

    if 'direction_accuracy' in metrics:
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Visualize results
    visualize_results(y_test, y_pred, feature_importance)

    return model, metrics, feature_importance


def visualize_results(y_test, y_pred, feature_importance):
    """
    Visualize model predictions and feature importance.
    """
    # Set up figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot actual vs predicted values
    axes[0].plot(y_test.values, label='Actual', color='blue', alpha=0.7)
    axes[0].plot(y_pred, label='Predicted', color='red', alpha=0.7)
    axes[0].set_title('Crude Oil Price Movement: Actual vs Predicted')
    axes[0].set_xlabel('Test Sample Index')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot feature importance
    top_n = min(10, len(feature_importance))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance.head(top_n),
        palette='viridis',
        ax=axes[1]
    )
    axes[1].set_title('Top 10 Feature Importance')
    axes[1].set_xlabel('Relative Importance (higher = more influential)')
    axes[1].set_ylabel('Feature Name')

    # Add annotation explaining importance values
    total_importance = feature_importance['Importance'].sum()
    top_importance = feature_importance.head(top_n)['Importance'].sum()
    axes[1].annotate(f'Top {top_n} features: {top_importance / total_importance:.1%} of total importance',
                     xy=(0.95, 0.05),
                     xycoords='axes fraction',
                     ha='right',
                     bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price Movement')
    plt.ylabel('Predicted Price Movement')
    plt.title('Actual vs Predicted Price Movement')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_xgboost_pipeline(df, sentiment_df):
    """
    Run the complete XGBoost pipeline for crude oil price prediction.
    """
    # Process data and add features
    processed_df = feature_preprocessing(df, sentiment_df)

    # Create, train and test XGBoost model
    model, metrics, feature_importance = create_and_train_model(processed_df)

    return model, processed_df, metrics


def main():
    price_data = pd.read_csv('../Local_Data/futures_full_30min_contin_UNadj_11assu1/CL_full_30min_continuous_UNadjusted.csv')
    sentiment_data = pd.read_csv('../Local_Data/crude_sentiments.csv')
    model, processed_df, metrics = run_xgboost_pipeline(price_data, sentiment_data)

if __name__ == '__main__':
    main()
