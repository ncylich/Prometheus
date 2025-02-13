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
    Replicates the PineScript Elliott Wave logic in Python,
    with additional validation rules. If a wave is invalid (e.g.,
    wave 4 overlapping wave 1), its label is removed.

    The resulting DataFrame has added columns:
       - 'sar': parabolic SAR values
       - 'wave_point': wave pivot point (NaN if no wave detected)
       - 'wave_num': which wave number (1..5, or 0 if invalid/none)
       - 'wave_dir': +1 (bullish) or -1 (bearish) or 0 (none)
       - 'fib_24': fib target for wave 2/4
       - 'fib_35': fib target for wave 3/5
       - 'prediction': naive guess of next wave direction
    """

    # 1) Compute Parabolic SAR
    df['sar'] = parabolic_sar(df['high'], df['low'], start, incre, maxim)

    # 2) Find SAR-close crossings
    df['x'] = df['sar'] - df['close']
    df['wave_bool'] = False  # Indicator for a wave pivot
    df['wave_point'] = np.nan

    for i in range(1, len(df)):
        prev_x = df['x'].iloc[i - 1]
        curr_x = df['x'].iloc[i]
        # If crossing from above (bullish pivot)
        if (prev_x > 0) and (curr_x < 0):
            df.at[df.index[i], 'wave_bool'] = True
            df.at[df.index[i], 'wave_point'] = (df['low'].iloc[i] + df['sar'].iloc[i]) / 2.0
        # If crossing from below (bearish pivot)
        elif (prev_x < 0) and (curr_x > 0):
            df.at[df.index[i], 'wave_bool'] = True
            df.at[df.index[i], 'wave_point'] = (df['high'].iloc[i] + df['sar'].iloc[i]) / 2.0

    # Build lists of wave pivot indexes and values for convenience.
    wave_indexes = df.index[df['wave_bool'] == True].tolist()
    wave_values = df.loc[df['wave_bool'] == True, 'wave_point'].tolist()

    def get_past_wave_value(idx, n_back):
        """Replicates PineScript's valuewhen(pivot, wave, n)."""
        if idx not in wave_indexes:
            return np.nan
        pos = wave_indexes.index(idx)
        wanted_pos = pos - n_back
        if wanted_pos < 0:
            return np.nan
        return wave_values[wanted_pos]

    # We'll store temporary wave labels here.
    w1 = np.zeros(len(df), dtype=int)
    w2 = np.zeros(len(df), dtype=int)
    w3 = np.zeros(len(df), dtype=int)
    w4 = np.zeros(len(df), dtype=int)
    w5 = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if not df['wave_bool'].iloc[i]:
            continue

        wave_i = df['wave_point'].iloc[i]
        # For clarity, define:
        # p1: pivot three waves ago (expected wave1 in a 4-wave sequence)
        # p2: pivot two waves ago (expected wave2)
        # p3: immediate previous pivot (expected wave3 when labeling wave4, or wave1 when labeling wave2)
        p1 = get_past_wave_value(df.index[i], 3)
        p2 = get_past_wave_value(df.index[i], 2)
        p3 = get_past_wave_value(df.index[i], 1)

        # ----- Wave 1 Logic -----
        # The first pivot in any sequence is taken as wave1.
        # (No extra validation is applied.)
        if np.isnan(p3):  # If there is no previous pivot, treat current as wave1.
            # Here we assign the sign based on price move direction (using SAR and close).
            w1[i] = 1 if df['close'].iloc[i] > df['sar'].iloc[i] else -1

        # ----- Wave 2 Logic -----
        # For a valid wave2, the retracement should not exceed 100% of wave1.
        # For bullish: current pivot (p2 candidate) must be higher than wave1 (p3).
        # For bearish: it must be lower.
        if (not np.isnan(p3)) and (not np.isnan(p2)):
            if (df['close'].iloc[i] > df['sar'].iloc[i]):  # bullish candidate
                if (p3 < wave_i):  # wave2 retracement should not go below wave1
                    w2[i] = 1
                else:
                    w2[i] = 0  # Invalidate if it retraces too far
            else:  # bearish candidate
                if (p3 > wave_i):
                    w2[i] = -1
                else:
                    w2[i] = 0

        # ----- Wave 3 Logic -----
        # For a valid wave3 in a bullish impulse, the new pivot must exceed both wave1 and wave2.
        # For bearish, it must be lower than both.
        if (not np.isnan(p3)) and (not np.isnan(p2)):
            # Retrieve the previous wave2 label (from the previous pivot)
            idx_in_waves = wave_indexes.index(df.index[i])
            if idx_in_waves > 0:
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w2 = w2[df.index.get_loc(prev_wp_idx)]
                if prev_w2 == 1 and (wave_i > p3) and (wave_i > p2):
                    w3[i] = 1
                elif prev_w2 == -1 and (wave_i < p3) and (wave_i < p2):
                    w3[i] = -1
                else:
                    w3[i] = 0  # Invalidate if rule not met

        # ----- Wave 4 Logic -----
        # In an impulse wave:
        # For bullish: wave4 should be a corrective move that does not “overlap” wave1,
        # meaning its value must be above wave1 (p1) and below wave3 (p3).
        # For bearish: the reverse.
        if (not np.isnan(p1)) and (not np.isnan(p3)):
            idx_in_waves = wave_indexes.index(df.index[i])
            if idx_in_waves > 0:
                # Look at the previous wave pivot’s wave3 label to know the impulse direction.
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w3 = w3[df.index.get_loc(prev_wp_idx)]
                if prev_w3 == 1:
                    # Bullish impulse: valid if p1 < wave_i < p3
                    if (wave_i > p1) and (wave_i < p3):
                        w4[i] = 1
                    else:
                        w4[i] = 0  # Invalidate overlapping wave4
                elif prev_w3 == -1:
                    # Bearish impulse: valid if p1 > wave_i > p3
                    if (wave_i < p1) and (wave_i > p3):
                        w4[i] = -1
                    else:
                        w4[i] = 0

        # ----- Wave 5 Logic -----
        # For a valid wave5:
        # In bullish: the new pivot should extend above wave3.
        # In bearish: it should extend below wave3.
        if not np.isnan(p3):
            idx_in_waves = wave_indexes.index(df.index[i])
            if idx_in_waves > 0:
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w4 = w4[df.index.get_loc(prev_wp_idx)]
                if prev_w4 == 1:
                    if wave_i > p3:
                        w5[i] = 1
                    else:
                        w5[i] = 0
                elif prev_w4 == -1:
                    if wave_i < p3:
                        w5[i] = -1
                    else:
                        w5[i] = 0

    # Combine waves 1..5 into final wave labeling.
    df['wave_num'] = 0
    df['wave_dir'] = 0

    for i in range(len(df)):
        w_num, w_dir = 0, 0
        # Priority: if wave5 is valid use that; if not, then wave4, etc.
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

    # --------- Fibonacci Levels (same as before) ----------
    df['fib_35'] = np.nan
    df['fib_24'] = np.nan

    for i in range(len(df)):
        if df['wave_bool'].iloc[i]:
            wave_i = df['wave_point'].iloc[i]
            wave_1_back = get_past_wave_value(df.index[i], 1)
            wave_2_back = get_past_wave_value(df.index[i], 2)

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
    df['prediction'] = None
    for i in range(len(df) - 1):
        wn = df['wave_num'].iloc[i]
        wd = df['wave_dir'].iloc[i]
        if wn == 5 and wd == 1:
            df.at[df.index[i], 'prediction'] = "Likely pullback (bearish)"
        elif wn == 5 and wd == -1:
            df.at[df.index[i], 'prediction'] = "Likely rally (bullish)"
        elif wn > 0:
            next_wave = wn + 1
            if next_wave > 5:
                next_wave = 1  # cycle back
            if wd == 1:
                df.at[df.index[i], 'prediction'] = f"Expect wave {next_wave} bullish"
            else:
                df.at[df.index[i], 'prediction'] = f"Expect wave {next_wave} bearish"

    return df


def detect_elliott_waves_relaxed(df,
                                 start=0.01,
                                 incre=0.01,
                                 maxim=0.1,
                                 show_all_waves=True,
                                 tol=0.01):
    """
    Detects Elliott waves with relaxed validation rules.

    Parameters:
      df: pd.DataFrame with columns: [open, high, low, close]
      start, incre, maxim: parameters for the Parabolic SAR.
      show_all_waves: if True, labels all waves (1 to 5).
      tol: tolerance (as a decimal fraction) to relax the strict validation rules.
           For example, tol=0.01 means a 1% deviation is allowed.

    Returns:
      The original df with added columns:
         - 'sar': parabolic SAR values
         - 'wave_point': detected pivot point
         - 'wave_num': wave number (1..5, or 0 if invalid/none)
         - 'wave_dir': direction of the wave (+1 for bullish, -1 for bearish)
         - 'prediction': a naive next-wave prediction
    """

    # 1) Compute Parabolic SAR
    df['sar'] = parabolic_sar(df['high'], df['low'], start, incre, maxim)

    # 2) Identify SAR-close crossings (potential pivots)
    df['x'] = df['sar'] - df['close']
    df['wave_bool'] = False
    df['wave_point'] = np.nan

    for i in range(1, len(df)):
        prev_x = df['x'].iloc[i - 1]
        curr_x = df['x'].iloc[i]
        if (prev_x > 0) and (curr_x < 0):
            df.at[df.index[i], 'wave_bool'] = True
            df.at[df.index[i], 'wave_point'] = (df['low'].iloc[i] + df['sar'].iloc[i]) / 2.0
        elif (prev_x < 0) and (curr_x > 0):
            df.at[df.index[i], 'wave_bool'] = True
            df.at[df.index[i], 'wave_point'] = (df['high'].iloc[i] + df['sar'].iloc[i]) / 2.0

    # Build lists of wave pivot indexes and values for convenience.
    wave_indexes = df.index[df['wave_bool'] == True].tolist()
    wave_values = df.loc[df['wave_bool'] == True, 'wave_point'].tolist()

    def get_past_wave_value(idx, n_back):
        """Retrieve the n-th previous pivot value."""
        if idx not in wave_indexes:
            return np.nan
        pos = wave_indexes.index(idx)
        wanted_pos = pos - n_back
        if wanted_pos < 0:
            return np.nan
        return wave_values[wanted_pos]

    # Temporary arrays for wave labels.
    w1 = np.zeros(len(df), dtype=int)
    w2 = np.zeros(len(df), dtype=int)
    w3 = np.zeros(len(df), dtype=int)
    w4 = np.zeros(len(df), dtype=int)
    w5 = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if not df['wave_bool'].iloc[i]:
            continue

        wave_i = df['wave_point'].iloc[i]
        # p3: immediate previous pivot (typically used as the base for comparisons)
        p3 = get_past_wave_value(df.index[i], 1)
        p2 = get_past_wave_value(df.index[i], 2)
        p1 = get_past_wave_value(df.index[i], 3)

        # ----- Wave 1 Logic -----
        # The very first detected pivot is considered wave1.
        if np.isnan(p3):
            w1[i] = 1 if df['close'].iloc[i] > df['sar'].iloc[i] else -1

        # ----- Wave 2 Logic -----
        # For bullish: instead of requiring wave_i > p3, allow a small tolerance.
        if (not np.isnan(p3)) and (not np.isnan(p2)):
            if df['close'].iloc[i] > df['sar'].iloc[i]:  # bullish candidate
                if wave_i > p3 * (1 - tol):  # relaxed condition
                    w2[i] = 1
                else:
                    w2[i] = 0
            else:  # bearish candidate
                if wave_i < p3 * (1 + tol):
                    w2[i] = -1
                else:
                    w2[i] = 0

        # ----- Wave 3 Logic -----
        # For bullish: wave_i should exceed both p3 and p2 by a tolerance factor.
        if (not np.isnan(p3)) and (not np.isnan(p2)):
            idx_in_waves = wave_indexes.index(df.index[i])
            if idx_in_waves > 0:
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w2 = w2[df.index.get_loc(prev_wp_idx)]
                if prev_w2 == 1:
                    if (wave_i > p3 * (1 + tol)) and (wave_i > p2 * (1 + tol)):
                        w3[i] = 1
                    else:
                        w3[i] = 0
                elif prev_w2 == -1:
                    if (wave_i < p3 * (1 - tol)) and (wave_i < p2 * (1 - tol)):
                        w3[i] = -1
                    else:
                        w3[i] = 0

        # ----- Wave 4 Logic -----
        # Allow wave4 to be a corrective move with some leniency.
        if (not np.isnan(p1)) and (not np.isnan(p3)):
            idx_in_waves = wave_indexes.index(df.index[i])
            if idx_in_waves > 0:
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w3 = w3[df.index.get_loc(prev_wp_idx)]
                if prev_w3 == 1:
                    # Bullish: relax the condition to allow slight deviations.
                    if (wave_i > p1 * (1 - tol)) and (wave_i < p3 * (1 + tol)):
                        w4[i] = 1
                    else:
                        w4[i] = 0
                elif prev_w3 == -1:
                    if (wave_i < p1 * (1 + tol)) and (wave_i > p3 * (1 - tol)):
                        w4[i] = -1
                    else:
                        w4[i] = 0

        # ----- Wave 5 Logic -----
        if not np.isnan(p3):
            idx_in_waves = wave_indexes.index(df.index[i])
            if idx_in_waves > 0:
                prev_wp_idx = wave_indexes[idx_in_waves - 1]
                prev_w4 = w4[df.index.get_loc(prev_wp_idx)]
                if prev_w4 == 1:
                    if wave_i > p3 * (1 + tol):
                        w5[i] = 1
                    else:
                        w5[i] = 0
                elif prev_w4 == -1:
                    if wave_i < p3 * (1 - tol):
                        w5[i] = -1
                    else:
                        w5[i] = 0

    # Combine wave labels to assign a final wave number and direction.
    df['wave_num'] = 0
    df['wave_dir'] = 0

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

    # Simple prediction logic remains similar.
    df['prediction'] = None
    for i in range(len(df) - 1):
        wn = df['wave_num'].iloc[i]
        wd = df['wave_dir'].iloc[i]
        if wn == 5 and wd == 1:
            df.at[df.index[i], 'prediction'] = "Likely pullback (bearish)"
        elif wn == 5 and wd == -1:
            df.at[df.index[i], 'prediction'] = "Likely rally (bullish)"
        elif wn > 0:
            next_wave = wn + 1
            if next_wave > 5:
                next_wave = 1
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



def plot_prediction_arrows(df, ax=None):
    """
    Plots the close price series along with arrows indicating the predictions.

    Up arrows (green) indicate a bullish prediction, while down arrows (red) indicate
    a bearish prediction. The function uses the 'wave_dir' column from the DataFrame.

    Parameters:
      df (pd.DataFrame): DataFrame containing at least the following columns:
                         'close', 'prediction', and 'wave_dir'
      ax (matplotlib.axes.Axes): Optional Axes object to plot on. If not provided,
                                 a new figure and axes are created.

    Returns:
      ax (matplotlib.axes.Axes): The Axes object with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the close price series
    ax.plot(df.index, df['close'], label='Close Price', color='black')

    # Create boolean masks for bullish and bearish predictions
    bullish_mask = df['prediction'].notnull() & (df['wave_dir'] > 0)
    bearish_mask = df['prediction'].notnull() & (df['wave_dir'] < 0)

    # Plot bullish predictions (up arrows)
    if bullish_mask.any():
        ax.scatter(df.index[bullish_mask], df['close'][bullish_mask],
                   marker='^', color='green', s=100, label='Bullish Prediction', zorder=5)

    # Plot bearish predictions (down arrows)
    if bearish_mask.any():
        ax.scatter(df.index[bearish_mask], df['close'][bearish_mask],
                   marker='v', color='red', s=100, label='Bearish Prediction', zorder=5)

    ax.set_title("Elliott Wave Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='best')

    return ax


def main_old():
    # -----------------------------
    # 1) Load Your Data
    # -----------------------------
    n_bars = 2000
    df_example = pd.read_parquet('../Local_Data/all_long_term_combo.parquet')
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


def plot_elliott_wave_pivots(df, ax=None):
    """
    Plots the close price series along with the Elliott Wave pivot points.
    Each valid pivot (where wave_num > 0) is marked with a unique marker/color
    and annotated with its wave number.

    Parameters:
      df (pd.DataFrame): DataFrame containing at least the columns:
                         'close', 'wave_point', 'wave_num'
      ax (matplotlib.axes.Axes): Optional. If provided, plots on this Axes.
                                  Otherwise, creates a new figure and axes.

    Returns:
      ax (matplotlib.axes.Axes): The Axes object with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the close price series
    ax.plot(df.index, df['close'], label="Close Price", color="black")

    # Define marker properties for each wave number
    wave_markers = {
        1: {'marker': 'o', 'color': 'blue', 'label': 'Wave 1'},
        2: {'marker': 's', 'color': 'green', 'label': 'Wave 2'},
        3: {'marker': '^', 'color': 'orange', 'label': 'Wave 3'},
        4: {'marker': 'v', 'color': 'purple', 'label': 'Wave 4'},
        5: {'marker': 'D', 'color': 'red', 'label': 'Wave 5'},
    }

    # For each wave number, plot the pivot points
    for wave_num, props in wave_markers.items():
        mask = df['wave_num'] == wave_num
        if mask.any():
            ax.scatter(df.index[mask], df['wave_point'][mask],
                       marker=props['marker'], color=props['color'],
                       s=100, label=props['label'], zorder=5)
            # Annotate each pivot with its wave number
            for idx in df.index[mask]:
                ax.annotate(str(wave_num),
                            (idx, df.loc[idx, 'wave_point']),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            color=props['color'])

    ax.set_title("Elliott Wave Pivot Points")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    return ax


def daily_arrow_counts(df):
    """
    Groups the DataFrame by day and counts the number of bullish and bearish arrows.

    Parameters:
      df (pd.DataFrame): DataFrame containing at least the 'wave_dir' column
                         and with a DateTimeIndex.

    Returns:
      pd.DataFrame: A DataFrame with one row per day and two columns:
                    'up_arrows' and 'down_arrows'.
    """
    # Create a copy and add a date column from the index
    df_daily = df.copy()
    df_daily['day'] = pd.to_datetime(df_daily.index).date

    # Group by date and count positive (up) and negative (down) wave directions
    up_counts = df_daily.groupby('day')['wave_dir'].apply(lambda x: (x > 0).sum())
    down_counts = df_daily.groupby('day')['wave_dir'].apply(lambda x: (x < 0).sum())

    # Combine into a single DataFrame
    daily_counts = pd.DataFrame({
        'up_arrows': up_counts,
        'down_arrows': down_counts
    })

    return daily_counts


if __name__ == "__main__":
    # -----------------------------
    # 1) Load and Prepare Your Data
    # -----------------------------
    try:
        df_example = pd.read_parquet('../Local_Data/all_long_term_combo.parquet')
    except Exception as e:
        raise Exception("Error loading data. Please ensure the parquet file exists and the path is correct.") from e

    # Rename columns for consistency
    df_example = df_example[['date', 'CL_close', 'CL_open', 'CL_high', 'CL_low']]
    df_example.set_index('date', inplace=True)
    # Use only a portion of the data for clarity
    # n_bars = 2000
    # df_example = df_example.tail(100) #.head(n_bars)
    df_example.columns = ['close', 'open', 'high', 'low']

    # -----------------------------
    # 2) Run the Updated Elliott Wave Detector
    # -----------------------------
    df_with_waves = detect_elliott_waves_relaxed(df_example, start=0.05, incre=0.05, maxim=0.5, tol=0.01)

    # (Optional) Print the last few rows to inspect wave labels and pivots
    print(df_with_waves[['close', 'wave_num', 'wave_dir', 'wave_point', 'prediction']].tail(20))

    plot = False
    if plot:
        # -----------------------------
        # 3) Plot Predictions and Wave Pivots
        # -----------------------------
        # Create a figure and axes to plot both predictions and pivot points on the same chart.
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot prediction arrows (green for bullish/up, red for bearish/down)
        ax = plot_prediction_arrows(df_with_waves, ax=ax)

        # Overlay Elliott Wave pivot points
        # ax = plot_elliott_wave_pivots(df_with_waves, ax=ax)

        plt.tight_layout()
        plt.show()

    # Calculate and print daily arrow counts
    counts = daily_arrow_counts(df_with_waves)
    print("Daily Arrow Counts:")
    print(counts)
    counts.to_csv("../Local_Data/daily_arrow_counts.csv")

