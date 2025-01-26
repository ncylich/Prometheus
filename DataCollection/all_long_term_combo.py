import os
import pandas as pd
from tqdm import tqdm


data_path = '../Local_Data/unique_focused_futures_30min'


def get_csvs(path=data_path):
    return [f for f in os.listdir(path) if f.endswith('.csv')]


def process_file(ticker, file):
    df = pd.read_csv(os.path.join(data_path, file))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')
    # rename columns to include ticker
    columns = {col: f"{ticker}_{col}" if col != 'date' else 'date' for col in df.columns}
    df = df.rename(columns=columns)
    return df


def naive_main():
    csvs = get_csvs()

    # Reorder so that CL is first
    cl_idx, cl_file = [(i, file) for i, file in enumerate(csvs) if file.startswith("CL_")][0]
    csvs = csvs[:cl_idx] + csvs[cl_idx + 1:]

    merged_df = process_file(cl_file.split('_')[0], cl_file)
    for file in tqdm(csvs):
        ticker = file.split('_')[0]
        df = process_file(ticker, file)
        merged_df = pd.merge(merged_df, df, on='date', how='inner')

    merged_df.to_parquet(os.path.join(data_path, 'all_long_term_combo.parquet'), index=False, compression='gzip')
    print(len(merged_df), "rows in the merged dataframe")


def main():
    csvs = get_csvs()

    # Reorder so that CL is first
    cl_idx, cl_file = [(i, file) for i, file in enumerate(csvs) if file.startswith("CL_")][0]
    csvs = csvs[:cl_idx] + csvs[cl_idx + 1:]

    merged_df = process_file(cl_file.split('_')[0], cl_file)
    for file in tqdm(csvs):
        ticker = file.split('_')[0]
        df = process_file(ticker, file)[['date', f'{ticker}_close', f'{ticker}_volume']]  # only keep close and volume
        merged_df = pd.merge(merged_df, df, on='date', how='left')

        merged_df[f'{ticker}_close'] = merged_df[f'{ticker}_close'].ffill()  # filling opens with previous close
        merged_df[f'{ticker}_volume'] = merged_df[f'{ticker}_volume'].fillna(0)  # filling in missing volumes with 0
        merged_df = merged_df.dropna()  # drop leading NaNs

    merged_df.to_parquet(os.path.join(data_path, 'interpolated_all_long_term_combo.parquet'), index=False, compression='gzip')
    print(len(merged_df), "rows in the merged dataframe")

if __name__ == "__main__":
    main()
