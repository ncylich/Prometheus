import glob
import os
from datetime import datetime
import pyarrow.parquet as pq
import pandas as pd

DATE = "20241030"

def retrieve_files(date):
    """
    Retrieve all parquet files for a given date
    """
    files = glob.glob(f"IB_Parquet_Data/{date}_output_*.parquet")
    cl_idx = files.index(f"IB_Parquet_Data/{date}_output_CL.parquet")  # making CL first
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

def update_df(df, ticker):
    """
    Update dataframe with ticker
    """
    # re-index
    df.reset_index(drop=False, inplace=True)

    df['expiry_and_date'] = df.apply(lambda row: f"{row['expiry']}.{row['date']}", axis=1)

    df = df.drop(columns=['dataMonth', 'expiry', 'date'])

    cols = set(df.columns) - {'expiry_and_date'}
    df = df.rename(columns={col: f"{ticker}_{col}" for col in cols})
    return df

def load_df_from_parquet(filename):
    """
    Load dataframe from parquet file
    """
    df = pq.read_table(filename).to_pandas()
    return df

def get_updated_dfs(files):
    """
    Get updated dataframes
    """
    dfs = []
    for file in files:
        df = load_df_from_parquet(file)
        ticker = get_ticker(file)
        df = update_df(df, ticker)
        dfs.append(df)
    return dfs

def main():
    files = retrieve_files(DATE)
    dfs = get_updated_dfs(files)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='expiry_and_date', how='inner')
    print(merged_df.columns)
    print(merged_df.head())
    print(len(merged_df))

    merged_df.reset_index(drop=False, inplace=True)

    merged_df['date'] = merged_df.apply(lambda row: row['expiry_and_date'].split('.')[1], axis=1)
    merged_df['expiry'] = merged_df.apply(lambda row: row['expiry_and_date'].split('.')[0], axis=1)
    merged_df.drop(columns=['expiry_and_date'], inplace=True)

    merged_df['date'] = pd.to_datetime(merged_df['date'], utc=True)
    merged_df['date'] = merged_df['date'].dt.tz_convert('America/New_York')

    # making date and expiry first and second columns
    cols = merged_df.columns.tolist()
    cols = cols[0:1] + cols[-2:] + cols[1:-2]
    merged_df = merged_df[cols]

    merged_df = merged_df.sort_values(by=['date', 'expiry'])

    merged_df.reset_index(inplace=True)
    merged_df.drop(columns=['level_0', 'index'], inplace=True)

    merged_df.to_csv(f"{DATE}_merged.csv", index=True)

if __name__ == '__main__':
    main()
