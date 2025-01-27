import glob
import os
from datetime import datetime
import pyarrow.parquet as pq
import pandas as pd

DATE = "20241111"
IGNORE_TICKERS = {'DX'}

def retrieve_files(date):
    """
    Retrieve all parquet files for a given date
    """
    files = glob.glob(f"IB_Processed_Data/{date}_output_*.parquet")
    cl_idx = files.index(f"IB_Processed_Data/{date}_output_CL.parquet")  # making CL first
    files = [files[cl_idx]] + files[:cl_idx] + files[cl_idx + 1:]
    return files


def get_ticker(filename):
    """
    Get ticker from filename
    :param filename:
    :return:
    """
    name = filename.split('.')[0]
    return name.split('_')[-1]

def update_df_col_names(df, ticker):
    """
    Update dataframe with ticker
    """
    cols = set(df.columns) - {'expiry', 'date'}
    df = df.rename(columns={col: f"{ticker}_{col}" for col in cols})
    return df

def get_updated_dfs(files):
    """
    Get updated dataframes
    """
    dfs = []
    for file in files:
        df = pq.read_table(file).to_pandas()
        ticker = get_ticker(file)
        if ticker in IGNORE_TICKERS:
            continue
        df = update_df_col_names(df, ticker)
        dfs.append(df)
    return dfs

def main(day=DATE):
    files = retrieve_files(day)
    dfs = get_updated_dfs(files)

    merged_df = dfs[0]
    for stock_df in dfs[1:]:
        merged_df = merged_df.merge(stock_df, on=['date', 'expiry'], how='inner')
    print(merged_df.columns)
    print(merged_df.head())
    print(len(merged_df))

    merged_df['date'] = pd.to_datetime(merged_df['date'], utc=True)
    merged_df['date'] = merged_df['date'].dt.tz_convert('America/New_York')

    def get_expiry_dist(row):
        expiry = row['expiry']
        year, month = expiry // 100, expiry % 100
        absolute_expiry = year * 12 + month

        date = row['date']
        year, month = date.year, date.month
        absolute_date = year * 12 + month

        return absolute_expiry - absolute_date

    merged_df['expiry-dist'] = merged_df.apply(get_expiry_dist, axis=1)
    cols = merged_df.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]  # putting new col, expiry-dist, 3rd
    merged_df = merged_df[cols]

    # Would sort, but it's already sorted

    merged_df.to_csv(f"{day}_merged.csv", index=True)

if __name__ == '__main__':
    main()
