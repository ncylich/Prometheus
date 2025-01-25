import os
import pandas as pd


data_path = '../Local_Data/focused_futures_30min'


def get_csvs(path=data_path):
    return [f for f in os.listdir(path) if f.endswith('.csv')]


def process_file(file):
    df = pd.read_csv(os.path.join(data_path, file))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')
    # rename columns to include ticker
    ticker = file.split('_')[0]
    columns = {col: f"{ticker}_{col}" if col != 'date' else 'date' for col in df.columns}
    df = df.rename(columns=columns)
    return df


def main():
    csvs = get_csvs()

    # Reorder so that CL is first
    cl_idx = [i for i, file in enumerate(csvs) if file.startswith("CL_")][0]
    csvs = [csvs[cl_idx]] + csvs[:cl_idx] + csvs[cl_idx + 1:]

    merged_df = process_file(csvs[0])
    for file in csvs[1:]:
        df = process_file(file)
        merged_df = pd.merge(merged_df, df, on='date', how='inner')

    merged_df.to_parquet(os.path.join(data_path, 'all_long_term_combo.parquet'), index=False, compression='gzip')
    print(len(merged_df), "rows in the merged dataframe")

if __name__ == "__main__":
    main()
