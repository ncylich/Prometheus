import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parabolic_sar(high, low, start=0.01, increment=0.01, maximum=0.1):
    """
    A basic Parabolic SAR implementation.

    :param high: pd.Series of highs
    :param low: pd.Series of lows
    :param start: acceleration factor start
    :param increment: acceleration factor increment
    :param maximum: acceleration factor max
    :return: pd.Series of SAR values
    """
    length = len(high)
    sar = np.zeros(length)
    # direction: +1 for uptrend, -1 for downtrend
    direction = 0
    af = start  # Acceleration Factor
    # extreme point (EP) = highest high (in an uptrend) or lowest low (in a downtrend)
    ep = 0
    # Initialize
    sar[0] = low[0] - (high[0] - low[0])  # rough init or pick something near first price
    direction = 1  # We'll guess up at first, can be refined
    ep = high[0]

    for i in range(1, length):
        # Step 1: Compute new SAR
        prev_sar = sar[i - 1]
        sar[i] = prev_sar + af * (ep - prev_sar)

        # Uptrend
        if direction == 1:
            if high[i] > ep:
                ep = high[i]
                af = min(af + increment, maximum)
            # Check if we switch from uptrend to downtrend
            if low[i] < sar[i]:
                direction = -1
                sar[i] = ep
                ep = low[i]
                af = start

        # Downtrend
        elif direction == -1:
            if low[i] < ep:
                ep = low[i]
                af = min(af + increment, maximum)
            # Check if we switch from downtrend to uptrend
            if high[i] > sar[i]:
                direction = 1
                sar[i] = ep
                ep = high[i]
                af = start

    return pd.Series(sar, index=high.index)


def detect_elliott_waves(df,
                         start=0.01,
                         incre=0.01,
                         maxim=0.1,
                         show_all_waves=True):
    """
    Faithfully replicate the PineScript Elliott Wave logic in Python.

    :param df: pd.DataFrame with columns: [open, high, low, close]
    :param start: float, initial parabolic SAR acceleration factor
    :param incre: float, parabolic SAR increment
    :param maxim: float, parabolic SAR acceleration factor max
    :param show_all_waves: bool, if True, keep track of all waves (1..5).
    :return: The original df with additional columns:
             - 'sar': parabolic SAR values
             - 'wave_point': wave pivot point (NaN if no wave detected)
             - 'wave_num': which wave number (1..5 or 0 if none)
             - 'wave_dir': +1 (bullish) or -1 (bearish) or 0 (none)
             - 'fib_24': fib target for wave 2/4
             - 'fib_35': fib target for wave 3/5
             - 'prediction': naive guess of next wave direction
    """

    # 1) Compute Parabolic SAR
    df['sar'] = parabolic_sar(df['high'], df['low'], start, incre, maxim)

    # 2) Replicate the logic:
    #    x = out - close
    #    we also want to check previous bar to see if sign changed.
    df['x'] = df['sar'] - df['close']

    # y = x[1] > 0 and x < 0  => crossing from above => wave pivot?
    # z = x[1] < 0 and x > 0  => crossing from below => wave pivot?
    # alpha = (y==true or z==true)
    # wave = if y => (low + out)/2 else if z => (high + out)/2
    df['wave_bool'] = False  # alpha
    df['wave_point'] = np.nan

    for i in range(1, len(df)):
        prev_x = df['x'].iloc[i - 1]
        curr_x = df['x'].iloc[i]
        if (prev_x > 0) and (curr_x < 0):
            # wave pivot y => crossing from above => typically price broke sar from above to below
            df.at[df.index[i], 'wave_bool'] = True
            df.at[df.index[i], 'wave_point'] = (df['low'].iloc[i] + df['sar'].iloc[i]) / 2.0
        elif (prev_x < 0) and (curr_x > 0):
            # wave pivot z => crossing from below => price broke sar from below to above
            df.at[df.index[i], 'wave_bool'] = True
            df.at[df.index[i], 'wave_point'] = (df['high'].iloc[i] + df['sar'].iloc[i]) / 2.0

    # For convenience, create a list of wave points (their index and value).
    wave_indexes = df.index[df['wave_bool'] == True].tolist()
    wave_values = df.loc[df['wave_bool'] == True, 'wave_point'].tolist()

    # Helper to replicate 'valuewhen(alfa, wave, n)' in PineScript:
    # We'll just find the n-th last wave pivot in wave_indexes, wave_values
    def get_past_wave_value(idx, n_back):
        # idx is the current bar index
        # we want to find the wave pivot that occurred n_back times ago
        # from the current wave pivot.
        # We'll see if the current bar 'idx' is also a wave pivot. If so, that is wave_indexes[-1].
        # We find where idx is in wave_indexes, then step back n_back.
        if idx not in wave_indexes:
            return np.nan

        pos = wave_indexes.index(idx)  # which wave pivot number is this bar
        wanted_pos = pos - n_back
        if wanted_pos < 0:
            return np.nan
        return wave_values[wanted_pos]

    # Next, define wave1, wave2, wave3, wave4, wave5 logic
    # We'll store final wave# in 'wave_num' column: 1..5 or 0 if none
    # We'll also store 'wave_dir': +1 for bullish wave, -1 for bearish wave, 0 if none
    df['wave_num'] = 0
    df['wave_dir'] = 0

    # We'll need some ephemeral columns in order to replicate logic exactly:
    # wave1, wave2, wave3, wave4, wave5 as in the script
    w1 = np.zeros(len(df), dtype=int)
    w2 = np.zeros(len(df), dtype=int)
    w3 = np.zeros(len(df), dtype=int)
    w4 = np.zeros(len(df), dtype=int)
    w5 = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if not df['wave_bool'].iloc[i]:
            continue

        wave_i = df['wave_point'].iloc[i]
        wave_1_back = get_past_wave_value(df.index[i], 1)
        wave_2_back = get_past_wave_value(df.index[i], 2)
        wave_3_back = get_past_wave_value(df.index[i], 3)

        # wave1 logic
        # wave1 = 1 if wave > BACK1
        # wave1 = -1 if wave < BACK1
        if not np.isnan(wave_1_back):
            if wave_i > wave_1_back:
                w1[i] = 1
            elif wave_i < wave_1_back:
                w1[i] = -1

        # wave2 logic
        # wave2=1 if BACK1>BACK2 and BACK1>wave and wave>BACK2
        # wave2=-1 if BACK1<BACK2 and BACK1<wave and wave<BACK2
        if not np.isnan(wave_1_back) and not np.isnan(wave_2_back):
            if (wave_1_back > wave_2_back
                    and wave_1_back > wave_i
                    and wave_i > wave_2_back):
                w2[i] = 1
            elif (wave_1_back < wave_2_back
                  and wave_1_back < wave_i
                  and wave_i < wave_2_back):
                w2[i] = -1

        # wave3 logic
        # wave3=1 if previous wave2=1 and wave>BACK1 and wave>BACK2
        # wave3=-1 if previous wave2=-1 and wave<BACK1 and wave<BACK2
        # We must check wave2 at the previous wave pivot (the wave pivot just prior to i)
        # That means we look for w2 at the index of the wave pivot that is "one wave pivot ago".
        # The wave2 is stored at that pivot index.
        if not np.isnan(wave_1_back) and not np.isnan(wave_2_back):
            # Which was the last wave pivot index? wave_indexes[-1]? Actually we do:
            idx_in_waves = wave_indexes.index(df.index[i])  # this wave pivot in wave_indexes
            if idx_in_waves > 0:
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w2 = w2[df.index.get_loc(prev_wp_idx)]  # wave2 at that pivot

                if (prev_w2 == 1
                        and wave_i > wave_1_back
                        and wave_i > wave_2_back):
                    w3[i] = 1
                elif (prev_w2 == -1
                      and wave_i < wave_1_back
                      and wave_i < wave_2_back):
                    w3[i] = -1

        # wave4 logic
        # wave4=1 if prev wave3=1 and wave<BACK1 and wave>BACK3
        # wave4=-1 if prev wave3=-1 and wave>BACK1 and wave<BACK3
        if not np.isnan(wave_3_back) and not np.isnan(wave_1_back):
            idx_in_waves = wave_indexes.index(df.index[i])
            if idx_in_waves > 0:
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w3 = w3[df.index.get_loc(prev_wp_idx)]
                if (prev_w3 == 1
                        and wave_i < wave_1_back
                        and wave_i > wave_3_back):
                    w4[i] = 1
                elif (prev_w3 == -1
                      and wave_i > wave_1_back
                      and wave_i < wave_3_back):
                    w4[i] = -1

        # wave5 logic
        # wave5=1 if prev wave4=1 and wave>BACK1
        # wave5=-1 if prev wave4=-1 and wave<BACK1
        if not np.isnan(wave_1_back):
            idx_in_waves = wave_indexes.index(df.index[i])
            if idx_in_waves > 0:
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w4 = w4[df.index.get_loc(prev_wp_idx)]
                if (prev_w4 == 1
                        and wave_i > wave_1_back):
                    w5[i] = 1
                elif (prev_w4 == -1
                      and wave_i < wave_1_back):
                    w5[i] = -1

    # Combine wave1..5 for final wave labeling: pick the largest wave # in priority (5 > 4 > 3 > 2 > 1)
    # Also set wave_dir to +1 or -1 depending on sign
    for i in range(len(df)):
        w_num, w_dir = 0, 0
        if w5[i] != 0:
            w_num = 5
            w_dir = w5[i]
        elif w4[i] != 0 and show_all_waves:
            w_num = 4
            w_dir = w4[i]
        elif w3[i] != 0 and show_all_waves:
            w_num = 3
            w_dir = w3[i]
        elif w2[i] != 0 and show_all_waves:
            w_num = 2
            w_dir = w2[i]
        elif w1[i] != 0 and show_all_waves:
            w_num = 1
            w_dir = w1[i]

        df.at[df.index[i], 'wave_num'] = w_num
        df.at[df.index[i], 'wave_dir'] = w_dir

    # --------- FIBONACCI LEVELS ----------
    # wave_35fib, wave_24fib from script
    df['fib_35'] = np.nan
    df['fib_24'] = np.nan

    for i in range(len(df)):
        if df['wave_bool'].iloc[i]:
            wave_i = df['wave_point'].iloc[i]
            wave_1_back = get_past_wave_value(df.index[i], 1)
            wave_2_back = get_past_wave_value(df.index[i], 2)

            # wave_35fib logic triggers if wave3=1 or wave5=1 or wave3=-1 or wave5=-1
            if (w3[i] == 1 or w5[i] == 1) and (not np.isnan(wave_1_back) and not np.isnan(wave_2_back)):
                a = wave_2_back - wave_1_back
                fibs = [
                    abs(wave_i - (wave_1_back + a * 1.618)),
                    abs(wave_i - (wave_1_back + a * 2.618)),
                    abs(wave_i - (wave_1_back + a * 3.618)),
                    abs(wave_i - (wave_1_back + a * 4.236))
                ]
                df.at[df.index[i], 'fib_35'] = wave_i + min(fibs)

            elif (w3[i] == -1 or w5[i] == -1) and (not np.isnan(wave_1_back) and not np.isnan(wave_2_back)):
                a = wave_1_back - wave_2_back
                fibs = [
                    abs(wave_i - (wave_1_back - a * 1.618)),
                    abs(wave_i - (wave_1_back - a * 2.618)),
                    abs(wave_i - (wave_1_back - a * 3.618)),
                    abs(wave_i - (wave_1_back - a * 4.236))
                ]
                df.at[df.index[i], 'fib_35'] = wave_i - min(fibs)

            # wave_24fib logic triggers if wave2=1 or wave4=1 or wave2=-1 or wave4=-1
            if (w2[i] == 1 or w4[i] == 1) and (not np.isnan(wave_1_back) and not np.isnan(wave_2_back)):
                a = wave_1_back - wave_2_back
                fibs = [
                    abs(wave_i - (wave_2_back + a * 0.786)),
                    abs(wave_i - (wave_2_back + a * 0.500)),
                    abs(wave_i - (wave_2_back + a * 0.382)),
                    abs(wave_i - (wave_2_back + a * 0.236))
                ]
                df.at[df.index[i], 'fib_24'] = wave_i + min(fibs)

            elif (w2[i] == -1 or w4[i] == -1) and (not np.isnan(wave_1_back) and not np.isnan(wave_2_back)):
                a = wave_2_back - wave_1_back
                fibs = [
                    abs(wave_i - (wave_2_back - a * 0.786)),
                    abs(wave_i - (wave_2_back - a * 0.500)),
                    abs(wave_i - (wave_2_back - a * 0.382)),
                    abs(wave_i - (wave_2_back - a * 0.236))
                ]
                df.at[df.index[i], 'fib_24'] = wave_i - min(fibs)

    # ---- Simple "Prediction" demonstration ----
    # If the last wave found is wave5 bullish, we guess a pullback next (bearish).
    # If the last wave found is wave5 bearish, we guess a rally next (bullish).
    # Otherwise, maybe we guess wave_n+1 is next in the same direction.
    df['prediction'] = None

    # We'll do a naive fill: for each bar, look at wave_num/wave_dir.
    # Then guess the next wave direction.
    for i in range(len(df) - 1):
        wn = df['wave_num'].iloc[i]
        wd = df['wave_dir'].iloc[i]
        if wn == 5 and wd == 1:
            df.at[df.index[i], 'prediction'] = "Likely pullback (bearish)"
        elif wn == 5 and wd == -1:
            df.at[df.index[i], 'prediction'] = "Likely rally (bullish)"
        elif wn > 0:
            # Just guess next wave in the same direction (this is overly simplistic)
            next_wave = wn + 1
            if next_wave > 5:
                next_wave = 1  # cycle
            if wd == 1:
                df.at[df.index[i], 'prediction'] = f"Expect wave {next_wave} bullish"
            else:
                df.at[df.index[i], 'prediction'] = f"Expect wave {next_wave} bearish"

    return df


def simple_elliott_predictor(df, wave_length=None):
    """
    A naive Elliott-based future price predictor.
    It:
      1) Finds the last confirmed wave pivot (wave_num in 1..5).
      2) Finds the pivot 'before it' (wave_num - 1 or the nearest prior wave pivot).
      3) Measures the time difference and price difference between those two pivots.
      4) Projects that same distance/amplitude forward from the last pivot.

    :param df: pd.DataFrame returned by detect_elliott_waves(...), containing:
               'wave_bool', 'wave_point', 'wave_num', 'wave_dir'
    :param wave_length: int or None
                       If None, we use the exact time gap from the previous wave pivot.
                       If an integer, we override the time gap with that many bars.
    :return: A new pd.DataFrame containing future date index and a column 'predicted_price'.
             If no wave pivot is found, returns an empty DataFrame.
    """
    # 1) Identify the index of the last wave pivot (where wave_num in 1..5).
    #    We'll also need the second-to-last wave pivot.
    wave_pivots = df[df['wave_num'] > 0].copy()
    if len(wave_pivots) < 2:
        # Not enough wave pivots to do a naive 'projection'
        return pd.DataFrame()

    # 2) The last wave pivot:
    last_pivot_idx = wave_pivots.index[-1]  # index of the last wave pivot
    last_pivot_bar = wave_pivots.loc[last_pivot_idx]

    # 3) The pivot before it:
    prev_pivot_idx = wave_pivots.index[-2]  # index of wave pivot before the last
    prev_pivot_bar = wave_pivots.loc[prev_pivot_idx]

    last_pivot_price = last_pivot_bar['wave_point']
    prev_pivot_price = prev_pivot_bar['wave_point']

    # 4) Compute time and price differences
    # Time difference: number of bars (or days if it's daily data) between prev_pivot_idx and last_pivot_idx
    dt = (last_pivot_idx - prev_pivot_idx)  # This could be a Timedelta if it's date-based
    n_bars = (df.index.get_loc(last_pivot_idx)
              - df.index.get_loc(prev_pivot_idx))  # difference in integer index positions

    price_diff = last_pivot_price - prev_pivot_price

    # 5) Decide how many future bars to project
    if wave_length is None:
        # Use the same number of bars as between the two pivots
        future_bars = n_bars
    else:
        future_bars = wave_length

    if future_bars < 1:
        # Not meaningful to do 0 or negative
        return pd.DataFrame()

    # 6) We'll create a future date index from the last_pivot_idx + 1 step up to future_bars steps
    #    If your index is daily, we'll assume you want daily steps.
    last_date = df.index[-1]
    # If the pivot index is also the last bar in df, we start from last_date + 1 freq step:
    freq_str = df.index.freqstr if df.index.freqstr else "D"  # fallback to daily

    # For a DatetimeIndex, we can do:
    future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq_str),
                                 periods=future_bars,
                                 freq=freq_str)

    # 7) We’ll do a *simple linear path* from the last pivot price to the new pivot:
    #    - next_pivot_price = last_pivot_price + price_diff
    #    - We'll distribute it across 'future_bars' steps
    next_pivot_price = last_pivot_price + price_diff
    step_size = (next_pivot_price - last_pivot_price) / future_bars

    # We'll produce a predicted_price that goes from [1..future_bars]
    preds = []
    current_price = last_pivot_price
    for i in range(future_bars):
        current_price += step_size
        preds.append(current_price)

    df_future = pd.DataFrame({'predicted_price': preds}, index=future_dates)

    return df_future


def sign(x):
    """
    A simple sign function:
     +1 if x > 0
      0 if x == 0
     -1 if x < 0
    """
    if x > 0: return 1
    if x < 0: return -1
    return 0


if __name__ == "__main__":

    # -----------------------------
    # 1) Load Your Data
    # -----------------------------
    n_bars = 2000
    df_example = pd.read_parquet('Local_Data/all_long_term_combo.parquet')
    # rename columns for convenience
    df_example = df_example[['date', 'CL_close', 'CL_open', 'CL_high', 'CL_low']]
    df_example.set_index('date', inplace=True)
    # Use only the last n_bars
    df_example = df_example.tail(6000).head(n_bars)
    # reorder to match detect_elliott_waves(...) expected columns
    df_example.columns = ['close', 'open', 'high', 'low']

    # -----------------------------
    # 2) Forecast Settings
    # -----------------------------
    step_size = 100  # every 100 bars, produce a new forecast
    forecast_horizon = 10  # predict exactly 10 bars forward

    # We'll store all forecasts for plotting
    predictions_list = []  # (train_end_loc, df_pred)

    # We'll track direction correctness in total
    correct_count = 0
    total_count = 0

    # -----------------------------
    # 3) Loop through the dataset
    # -----------------------------
    # For each iteration:
    #   - train on the first `train_end_loc` bars
    #   - predict the next `forecast_horizon` bars
    #   - check final direction correctness
    for train_end_loc in range(step_size, len(df_example), step_size):
        # a) Prepare the "training" data portion
        df_train = df_example.iloc[:train_end_loc].copy()

        # b) Detect Elliott waves on training portion
        train_result = detect_elliott_waves(
            df_train,
            start=0.01,
            incre=0.01,
            maxim=0.1
        )

        # c) Predict future for exactly `forecast_horizon` bars
        df_pred = simple_elliott_predictor(
            train_result,
            wave_length=forecast_horizon
        )

        # If no pivot or empty result, we skip
        if df_pred.empty:
            predictions_list.append((train_end_loc, df_pred))
            continue

        predictions_list.append((train_end_loc, df_pred))

        # d) Compare the final predicted bar's direction vs. actual
        #    Direction = sign( final_pred - last_train_close )

        # The last bar in the training portion:
        last_train_close = df_train['close'].iloc[-1]

        # The final bar in the forecast:
        final_pred_date = df_pred.index[-1]
        predicted_final_price = df_pred['predicted_price'].iloc[-1]

        # Predicted direction from training-end to final forecast bar
        pred_dir = sign(predicted_final_price - last_train_close)

        # Actual direction: if we do have the final_pred_date in our df_example,
        # we can check the actual close. If not, we skip counting.
        if final_pred_date in df_example.index:
            actual_final_close = df_example.loc[final_pred_date, 'close']
            act_dir = sign(actual_final_close - last_train_close)

            # If directions match (including 0->0), we call it correct
            if pred_dir == act_dir:
                correct_count += 1

            total_count += 1

    # ---------------------------------------
    # 4) Final overall direction proportion
    # ---------------------------------------
    if total_count > 0:
        overall_direction_accuracy = correct_count / total_count
        print(f"Overall direction correctness: {overall_direction_accuracy:.2%}")
    else:
        print("No valid predictions to compare.")

    # ---------------------------------------
    # 5) Plot all predictions
    # ---------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the entire actual close in black
    ax.plot(df_example.index, df_example['close'], label='Actual Close', color='black')

    # Over-plot all predictions in pink
    first_label = True
    for (train_end_loc, df_pred) in predictions_list:
        if not df_pred.empty:
            lbl = 'Predictions' if first_label else '_nolegend_'
            ax.plot(df_pred.index, df_pred['predicted_price'],
                    color='magenta', linewidth=1.5, alpha=0.6,
                    label=lbl)
            first_label = False

    ax.set_title(f"Elliott Wave Predictions Every ~{step_size} Bars\n"
                 f"Forecast Horizon = {forecast_horizon} bars\n"
                 f"Direction Accuracy = {overall_direction_accuracy:.2%}"
                 if total_count > 0 else "")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()