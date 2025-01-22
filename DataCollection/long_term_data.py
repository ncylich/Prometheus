import os
import gc
import csv
import sys
import tempfile
import pandas as pd
import pyarrow as pq
from tqdm import tqdm


COLS = {
    'date': 'str',
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
}

fill_times = True
unadjusted = True

n_min = 30
unadjusted_postfix = f'_full_{n_min}min_continuous_UNadjusted'
adjusted_postfix = f'_full_{n_min}min_continuous_ratio_adjusted'

# data_dir = 'long_term_data'
# result_dir = '../Local_Data'
data_dir = '../Local_Data/futures_full_30min_contin_UNadj_11assu1'
result_dir = '../Local_Data'


def get_files(path=data_dir):
    """
    Get all files in a directory
    """
    all_files = os.listdir(path)
    data_files = [f for f in all_files if unadjusted_postfix in f or adjusted_postfix in f]
    data_files = [f for f in data_files if not f.endswith('_filled.csv')]  # remove filled files
    data_files = add_cols(data_files)
    return data_files

def add_cols(data_files):
    '''
    Prepares the data files by adding the columns to the files that are missing them and correcting ending
    '''
    cols = ','.join(COLS.keys()) + '\n'
    new_files = []
    for f in data_files:
        if not f.endswith('.txt'):
            new_files.append(f)
            continue
        rename_only = True
        with open(os.path.join(data_dir, f), 'r') as file:
            lines = file.readlines()
            if lines[0] != cols:
                lines = [cols] + lines
                rename_only = False

        new_file_name = f[:-4] + '.csv'
        if rename_only:
            os.rename(os.path.join(data_dir, f), os.path.join(data_dir, new_file_name))
        else:
            with open(os.path.join(data_dir, new_file_name), 'w') as file:
                file.writelines(lines)
            os.remove(os.path.join(data_dir, f))
        new_files.append(new_file_name)

    return new_files

def get_ticker(file):
    """
    Get ticker from file
    """
    return file.split('_')[0]

def get_tickers(files):
    """
    Get tickers from files
    """
    tickers = set()
    for file in files:
        tickers.add(get_ticker(file))
    return tickers

def fill_missing_times_with_file(df, filled_file):
    # Ensure the data is sorted by date
    df = df.sort_values(by='date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')
    cols = {col: f"{col.strip()}" for col in df.columns}
    df = df.rename(columns=cols)

    def fill_data_times(start_fill, end_fill, price, include_header=False):
        missing_times = pd.date_range(
            start=start_fill,
            end=end_fill,
            freq='min'
        )

        if len(missing_times) > 0:
            pd.DataFrame({
                'date': missing_times,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0
            }).to_csv(filled_file, mode='a', index=False, header=include_header)

    def fill_start_of_day(row, include_header=False):
        start_delta = row['date'].minute % 30
        if start_delta != 0:
            start_time = row['date'] - pd.Timedelta(minutes=start_delta)
            end_time = row['date'] - pd.Timedelta(minutes=1)
            fill_data_times(start_time, end_time, row['open'], include_header=include_header)
            return True
        return False

    def fill_end_of_day(row):
        end_delta = 30 - row['date'].minute % 30
        if end_delta != 30:
            start_time = row['date'] + pd.Timedelta(minutes=1)
            end_time = row['date'] + pd.Timedelta(minutes=end_delta)
            fill_data_times(start_time, end_time, row['close'])
            return True
        return False

    prev_row = df.iloc[0]
    prev_date = prev_row['date']

    header_exists = fill_start_of_day(prev_row, include_header=True)
    df.head(1).to_csv(filled_file, mode='a', index=False, header=not header_exists)

    for i in tqdm(range(1, len(df))):
        current_row = df.iloc[i]
        curr_date = current_row['date']

        if prev_date.day == curr_date.day:  # Same day
            fill_data_times(prev_date + pd.Timedelta(minutes=1), curr_date - pd.Timedelta(minutes=1), prev_row['close'])
        else:
            fill_end_of_day(prev_row)
            fill_start_of_day(current_row)

        # faster than using pd to save
        with open(filled_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(current_row)
        # pd.DataFrame([current_row]).to_csv(filled_file, mode='a', index=False, header=not os.path.exists(filled_file))

        prev_row = current_row
        prev_date = curr_date

    fill_end_of_day(prev_row)

    del df

def fill_missing_times(df):
    # Ensure the data is sorted by date
    df = df.sort_values(by='date').reset_index(drop=True)
    results = [df.head(1).copy()]  # Start with the first row

    prev_row = df.iloc[0]
    for i in tqdm(range(1, len(df))):
        current_row = df.iloc[i]

        prev_date = prev_row['date']
        curr_date = current_row['date']

        # Only fill gaps if both rows are on the same calendar day
        # (replicating the original code's conditional logic)
        if prev_date.day == curr_date.day:  # Same day (ignores month, year, etc.)
            # Compute the missing times between prev_date and curr_date
            # We start from prev_date + 1 minute and go up to curr_date - 1 minute
            missing_times = pd.date_range(
                start=prev_date + pd.Timedelta(minutes=1),
                end=curr_date - pd.Timedelta(minutes=1),
                freq='min'
            )

            if len(missing_times) > 0:
                # Vectorized construction of missing rows
                # For these rows:
                # - open = prev_row['close']
                # - high = max(prev_row['close'], current_row['open'])
                # - low = min(prev_row['close'], current_row['open'])
                # - close = current_row['open']
                # - volume = 0
                missing_data = pd.DataFrame({
                    'date': missing_times,
                    'open': prev_row['close'],
                    'high': max(prev_row['close'], current_row['open']),
                    'low': min(prev_row['close'], current_row['open']),
                    'close': current_row['open'],
                    'volume': 0
                })
                results.append(missing_data)

        # Append the current row
        results.append(pd.DataFrame(current_row))
        prev_row = current_row

    # Concatenate everything into a single DataFrame
    new_df = pd.concat(results, ignore_index=True)
    return new_df


def load_df(file):
    assert file.endswith('.csv'), f'File type not supported for: {file}'
    assert os.path.exists(file), f'File not found: {file}'

    if fill_times:
        filled_file = file.replace('.csv', '_filled.csv')
        if os.path.exists(filled_file):
            df = pd.read_csv(filled_file)
        else:
            df = pd.read_csv(file, dtype=COLS) if file.endswith('.csv') else pd.read_parquet(file)
            try:
                fill_missing_times_with_file(df, filled_file)  # deletes df after use to save memory
            except Exception as e:
                os.remove(filled_file)
                raise e
            df = pd.read_csv(filled_file)
    else:
        df = pd.read_csv(file, dtype=COLS) if file.endswith('.csv') else pd.read_parquet(file)
        cols = {col: f"{col.strip()}" for col in df.columns}
        df = df.rename(columns=cols)

    # df['date'] = pd.to_datetime(df['date'], utc=True)
    # df['date'] = df['date'].dt.tz_convert('America/New_York')

    return df

def load_and_format_df(file):
    """
    Load dataframe from file
    """
    df = load_df(os.path.join(data_dir, file))

    ticker = get_ticker(file)
    cols = set(df.columns) - {'date'}
    cols = {col: f"{ticker}_{col}" for col in cols}
    df = df.rename(columns=cols)

    return df

def merge_data(data_files, adjusted=False):
    """
    Merge data files
    """
    if adjusted:
        data_files = [f for f in data_files if adjusted_postfix in f]
    else:
        data_files = [f for f in data_files if unadjusted_postfix in f]

    # CL is the first file and first cols in the merged dataframe
    cl_str = [f for f in data_files if 'CL' in f][0]
    cl_idx = data_files.index(cl_str)
    data_files = [data_files[cl_idx]] + data_files[:cl_idx] + data_files[cl_idx + 1:]

    merged_df = load_and_format_df(data_files[0])
    for file in tqdm(data_files[1:]):
        stock_df = load_and_format_df(file)
        print(f'\n{file.split("_")[0]}: {len(stock_df): ,} rows')
        # merged_df = merged_df.merge(stock_df, on=['date'], how='inner')
        del stock_df
        gc.collect()

    return merged_df

def run():
    files = get_files()

    if unadjusted:
        df = merge_data(files)
        print('UNadjusted number of row:', len(df))
        path = os.path.join(result_dir, f'All_30min_long_term_merged_UNadjusted.parquet')
        df.to_parquet(path, compression='snappy', index=False)
    else:
        df = merge_data(files, adjusted=True)
        print('adjusted number of row:', len(df))
        path = os.path.join(result_dir, f'1min_long_term_merged_adjusted.parquet')
        df.to_parquet(path, compression='snappy', index=False)

def clean_old_filled_files():
    files = os.listdir(data_dir)
    for f in files:
        if f.endswith('_filled.csv'):
            os.remove(os.path.join(data_dir, f))

def main(path: str = '', res_dir: str = '') -> None:
    start = pd.Timestamp.now()
    # clean_old_filled_files()
    run()
    print('Done:', pd.Timestamp.now() - start)

if __name__ == '__main__':
    main()
