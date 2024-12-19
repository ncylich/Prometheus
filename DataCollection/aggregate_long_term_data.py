import time

import long_term_data
import pandas as pd
import sys
import os


def x_minute_file_name(x_min):
    return f'{x_min}min_long_term_merged_UNadjusted.parquet'
file = x_minute_file_name(1)

try:
    data = pd.read_parquet(file)
except FileNotFoundError:
    print(f'File not found, generating with long_term_data.py: {file}')
    long_term_data.main()
    time.sleep(1)
    data = pd.read_parquet(file)

def get_tickers(df):
    all_tickers = [col.split('_')[0] for col in df.columns]
    tickers_set = set(all_tickers)
    ordered_tickers = []
    for ticker in all_tickers:
        if ticker in tickers_set:
            ordered_tickers.append(ticker)
            tickers_set.remove(ticker)
    return ordered_tickers
tickers = get_tickers(data)

def aggregate_long_term_data(interval: int=5):
    df = pd.DataFrame(data.head(0))
    for i in range(0, len(data), interval):
        # set initial values to first row
        row = data.iloc[i]
        values = {col: row[col] for col in df.columns}
        for j in range(1, interval):
            row = data.iloc[i+j]
            for col in df.columns:
                if 'volume' in col:
                    values[col] += row[col]  # sums volume
                elif 'close' in col:
                    values[col] = row[col]  # takes final close price
                elif 'high' in col:
                    values[col] = max(values[col], row[col])  # takes max high price
                elif 'low' in col:
                    values[col] = min(values[col], row[col])  # takes min low price
                elif 'date' in col or 'open' in col:
                    continue  # skips date and open columns as they are derived from the original row
                else:
                    raise ValueError(f'Column name not recognized: {col}')

        ordered_values = [values[col] for col in df.columns]  # orders values to match df
        df.loc[len(df)] = ordered_values  # appends row to df

if __name__ == '__main__':
    time_interval = 5
    result = aggregate_long_term_data(time_interval)
    result.to_parquet(x_minute_file_name(time_interval))
