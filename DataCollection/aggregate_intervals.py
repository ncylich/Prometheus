import pandas as pd
from torch import dtype
import long_term_data
from tqdm import tqdm
import tempfile

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


def aggregate_intervals(df, interval=5):
    """
    Aggregate data into intervals:
    - outputs to temp file with csv format initially to avoid memory and slow down issues
    """
    cols = df.columns
    with tempfile.NamedTemporaryFile(delete=True, suffix='.csv', mode='w+') as temp_file:
        temp_file.write(','.join(cols) + '\n')
        for i in tqdm(range(0, len(df) - (interval - 1), interval)):
            int_rows = df.iloc[i: i + interval]
            new_row = [process_col(int_rows[col], col) for col in cols]
            new_line = ','.join([str(x) for x in new_row]) + '\n'
            temp_file.write(new_line)
        temp_file.seek(0)
        new_df = long_term_data.load_df(temp_file.name)
    return new_df


def main():
    print('Aggregating UNadjusted 1 min data into 5 min intervals')

    # loading parquet
    df = long_term_data.load_df('long_term_data/1min_long_term_merged_UNadjusted.parquet')
    df = aggregate_intervals(df)
    df.to_parquet('long_term_data/5min_long_term_merged_UNadjusted.parquet', compression='snappy', index=False)

    print('Aggregating adjusted 1 min data into 5 min intervals')
    df = long_term_data.load_df('long_term_data/1min_long_term_merged_adjusted.parquet')
    df = aggregate_intervals(df)
    df.to_parquet('long_term_data/5min_long_term_merged_adjusted.parquet', compression='snappy', index=False)

if __name__ == '__main__':
    main()
