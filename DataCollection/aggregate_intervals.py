import os

import pandas as pd
import long_term_data
from tqdm import tqdm

'''
Given a df of the form: date,CL_open,CL_high,CL_low,CL_close,CL_volume,SI_open,SI_high,SI_low,SI_close,SI_volume,NG_open,NG_high,NG_low,NG_close,NG_volume,GC_open,GC_high,GC_low,GC_close,GC_volume,HG_open,HG_high,HG_low,HG_close,HG_volume,DX_open,DX_high,DX_low,DX_close,DX_volume,VX_open,VX_high,VX_low,VX_close,VX_volume,ZN_open,ZN_high,ZN_low,ZN_close,ZN_volume

With 1 min intervals for each row, aggregate the data into 5 minute intervals.
'''


def process_col(col, col_name):
    if 'date' in col_name:
        return col.iloc[0]
    if 'open' in col_name:
        return col.iloc[0]
    if 'high' in col_name:
        return col.max()
    if 'low' in col_name:
        return col.min()
    if 'close' in col_name:
        return col.iloc[-1]
    if 'volume' in col_name:
        return col.sum()


def aggregate_intervals(df, saved_name, interval=5):
    """
    Aggregate data into intervals:
    - outputs directly into a csv instead of df to save memory and time for large datasets
    """
    cols = df.columns
    saved_csv = saved_name + '.csv'
    with open(saved_csv, 'w') as f:
        f.write(','.join(cols) + '\n')
        for i in tqdm(range(0, len(df) - (interval - 1), interval)):
            int_rows = df.iloc[i: i + interval]
            new_row = [process_col(int_rows[col], col) for col in cols]
            new_line = ','.join([str(x) for x in new_row]) + '\n'
            f.write(new_line)

    df = pd.read_csv(saved_csv)
    saved_parquet = saved_name + '.parquet'
    df.to_parquet(saved_parquet, compression='snappy', index=False)
    os.remove(saved_csv)


def main():
    print('Aggregating UNadjusted 1 min data into 5 min intervals')
    #df = pd.read_csv('long_term_data/1min_long_term_merged_UNadjusted.csv', dtype=long_term_data.COLS)

    # loading parquet
    df = pd.read_parquet('long_term_data/1min_long_term_merged_UNadjusted.parquet')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')
    aggregate_intervals(df, 'long_term_data/5_min_long_term_merged_UNadjusted')

    print('Aggregating adjusted 1 min data into 5 min intervals')
    df = pd.read_parquet('long_term_data/1min_long_term_merged_adjusted.parquet', dtype=long_term_data.COLS)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')
    aggregate_intervals(df, 'long_term_data/5min_long_term_merged_adjusted')

if __name__ == '__main__':
    main()
