from tqdm import tqdm
import long_term_data
import pandas as pd
import numpy as np
import time
import os

UNADJUSTED = True
TIME_INTERVAL = 60


def x_minute_file_name(x_min):
    file_name = f'{x_min}min_long_term_merged_{"UN" if UNADJUSTED else ""}adjusted.parquet'
    return os.path.join('..', 'Local_Data', file_name)

def get_data():
    file = x_minute_file_name(1)
    try:
        data = pd.read_parquet(file)
    except FileNotFoundError:
        print(f'File not found, generating with long_term_data.py: {file}')
        long_term_data.main()
        time.sleep(1)
        data = pd.read_parquet(file)
    return data


# def get_tickers(df):
#     all_tickers = [col.split('_')[0] for col in df.columns]
#     tickers_set = set(all_tickers)
#     ordered_tickers = []
#     for ticker in all_tickers:
#         if ticker in tickers_set:
#             ordered_tickers.append(ticker)
#             tickers_set.remove(ticker)
#     return ordered_tickers
# tickers = get_tickers(data)


def aggregate_long_term_data(data, interval: int=5):
    df = pd.DataFrame(data.head(0))
    for i in tqdm(range(0, len(data), interval)):
        # set initial values to first row
        row = data.iloc[i]
        values = [row[col] for col in df.columns]
        end_idx = min(i + interval, len(data))
        for j in range(i + 1, end_idx):
            row = data.iloc[j]
            for col_idx, col in enumerate(df.columns):
                if 'volume' in col:
                    values[col_idx] += row[col]  # sums volume
                elif 'close' in col:
                    values[col_idx] = row[col]  # takes final close price
                elif 'high' in col:
                    values[col_idx] = max(values[col_idx], row[col])  # takes max high price
                elif 'low' in col:
                    values[col_idx] = min(values[col_idx], row[col])  # takes min low price
                elif 'date' in col or 'open' in col:
                    continue  # skips date and open columns as they are derived from the original row
                else:
                    raise ValueError(f'Column name not recognized: {col}')

        df.loc[len(df)] = values  # appends row to df

    return df


def optimized_aggregate_long_term_data(data, interval: int = 5) -> pd.DataFrame:
    """
    Aggregates data in chunks of `interval` rows.

    Rules (based on original code logic):
      - 'volume' columns: sum of volumes
      - 'close' columns: final (last) close
      - 'high' columns: max high
      - 'low' columns: min low
      - 'date' or 'open' columns: first row in the chunk
      - Otherwise: raise ValueError
    """

    # Build a dictionary that tells Pandas how to aggregate each column
    aggregator = {}
    for col in data.columns:
        if 'volume' in col:
            aggregator[col] = 'sum'
        elif 'close' in col:
            aggregator[col] = 'last'
        elif 'high' in col:
            aggregator[col] = 'max'
        elif 'low' in col:
            aggregator[col] = 'min'
        elif 'date' in col or 'open' in col:
            aggregator[col] = 'first'
        else:
            raise ValueError(f'Column name not recognized for aggregation: {col}')

    # Create a "group" identifier for each row, e.g. rows 0..4 -> group 0, rows 5..9 -> group 1, etc.
    groups = np.arange(len(data)) // interval
    data['group'] = groups

    # Perform the groupby-aggregation
    df_agg = data.groupby('group', as_index=False).agg(aggregator)

    # Drop the group column
    df_agg.drop(columns='group', inplace=True)

    # Optional: rename the grouping column if you want a clean DataFrame index
    # Here, we can drop the group column altogether after aggregation:
    return df_agg.reset_index(drop=True)


def main(time_interval):
    data = get_data()
    # result = aggregate_long_term_data(data, time_interval)
    result = optimized_aggregate_long_term_data(data, time_interval)
    result.to_parquet(x_minute_file_name(time_interval), compression='snappy', index=False)
    print(result)


if __name__ == '__main__':
    # main(TIME_INTERVAL)
    for n in (5, 30, 60):
        main(n)
