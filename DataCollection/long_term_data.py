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

unadjusted_postfix = '_full_1min_continuous_UNadjusted'
adjusted_postfix = '_full_1min_continuous_ratio_adjusted'
data_dir = 'long_term_data'
result_dir = '../Local_Data'

def get_files(path=data_dir):
    """
    Get all files in a directory
    """
    all_files = os.listdir(path)
    data_files = [f for f in all_files if unadjusted_postfix in f or adjusted_postfix in f]
    data_files = [f for f in data_files if not f.endswith('_filled.csv')]
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

# def fill_missing_times(df):
#     new_df = df.head(0).copy()
#     start_date = df.iloc[0]['date']
#     prev_row = df.iloc[0]
#     for i in tqdm(range(len(df))):
#         row = df.iloc[i]
#         date = row['date']
#
#         # if same day
#         if date.day == start_date.day:
#             start_date += pd.Timedelta(minutes=1)  # start from the next minute
#             if start_date != date:
#                 new_row = {'date': start_date, 'open': prev_row['close'], 'high': max(prev_row['close'], row['open']),
#                            'low': min(prev_row['close'], row['open']), 'close': row['open'], 'volume': 0}
#                 new_row = [new_row[col] for col in new_df.columns]
#                 while start_date < date:
#                     new_df.loc[len(new_df)] = new_row
#                     start_date += pd.Timedelta(minutes=1)
#
#         new_df.loc[len(new_df)] = row
#         prev_row = row
#         start_date = date
#     return new_df

# def fill_missing_times_with_temp_file(df):
#     # Ensure the data is sorted by date
#     df = df.sort_values(by='date').reset_index(drop=True)
#
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
#         temp_file = temp_file.name
#
#     chunk_size = 10000  # Process in chunks to manage memory, 10,000
#
#     prev_row = df.iloc[0]
#     results = [df.head(1).copy()]  # Temporary buffer for rows
#     for i in tqdm(range(1, len(df))):
#         current_row = df.iloc[i]
#         prev_date = prev_row['date']
#         curr_date = current_row['date']
#
#         if prev_date.day == curr_date.day:  # Same day
#             missing_times = pd.date_range(
#                 start=prev_date + pd.Timedelta(minutes=1),
#                 end=curr_date - pd.Timedelta(minutes=1),
#                 freq='min'
#             )
#
#             if len(missing_times) > 0:
#                 missing_data = pd.DataFrame({
#                     'date': missing_times,
#                     'open': prev_row['close'],
#                     'high': max(prev_row['close'], current_row['open']),
#                     'low': min(prev_row['close'], current_row['open']),
#                     'close': current_row['open'],
#                     'volume': 0
#                 })
#                 results.append(missing_data)
#
#         results.append(pd.DataFrame([current_row]))
#
#         # Periodically write to temp file
#         if len(results) >= chunk_size:
#             pd.concat(results).to_csv(temp_file, mode='a', index=False, header=not os.path.exists(temp_file))
#             results = []  # Clear the buffer
#
#         prev_row = current_row
#
#     # Write any remaining rows
#     if results:
#         pd.concat(results).to_csv(temp_file, mode='a', index=False, header=not os.path.exists(temp_file))
#
#     # Read the temp file back into a DataFrame
#     new_df = pd.read_csv(temp_file)
#     os.remove(temp_file)  # Clean up the temp file
#     return new_df

def fill_missing_times_with_file(df, filled_file):
    # Ensure the data is sorted by date
    df = df.sort_values(by='date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')
    cols = {col: f"{col.strip()}" for col in df.columns}
    df = df.rename(columns=cols)

    prev_row = df.iloc[0]
    prev_date = prev_row['date']

    # Adding in empty spots at the beginning of the day in case the first row doesn't start at 9:30
    start_delta = prev_date.minute % 30
    if start_delta != 0:
        missing_times = pd.date_range(
            start=prev_date - pd.Timedelta(minutes=start_delta),
            end=prev_date - pd.Timedelta(minutes=1),
            freq='min'
        )

        price = prev_row['open']
        pd.DataFrame({
            'date': missing_times,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0
        }).to_csv(filled_file, mode='a', index=False, header=True)

    df.head(1).to_csv(filled_file, mode='a', index=False, header=(start_delta == 0))

    for i in tqdm(range(1, len(df))):
        current_row = df.iloc[i]
        curr_date = current_row['date']

        if prev_date.day == curr_date.day:  # Same day
            missing_times = pd.date_range(
                start=prev_date + pd.Timedelta(minutes=1),
                end=curr_date - pd.Timedelta(minutes=1),
                freq='min'
            )

            if len(missing_times) > 0:
                missing_data = pd.DataFrame({
                    'date': missing_times,
                    'open': prev_row['close'],
                    'high': max(prev_row['close'], current_row['open']),
                    'low': min(prev_row['close'], current_row['open']),
                    'close': current_row['open'],
                    'volume': 0
                })
                missing_data.to_csv(filled_file, mode='a', index=False, header=False)

        # faster than using pd to save
        with open(filled_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(current_row)
        # pd.DataFrame([current_row]).to_csv(filled_file, mode='a', index=False, header=not os.path.exists(filled_file))

        prev_row = current_row
        prev_date = curr_date

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

    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')

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
        merged_df = merged_df.merge(stock_df, on=['date'], how='inner')
        del stock_df
        gc.collect()

    return merged_df

def main():
    files = get_files()

    if unadjusted:
        df = merge_data(files)
        print('UNadjusted number of row:', len(df))
        path = os.path.join(result_dir, f'1min_long_term_merged_UNadjusted{"_filled" if fill_times else ""}.parquet')
        df.to_parquet(path, compression='snappy', index=False)
    else:
        df = merge_data(files, adjusted=True)
        print('adjusted number of row:', len(df))
        path = os.path.join(result_dir, f'1min_long_term_merged_adjusted{"_filled" if fill_times else ""}.parquet')
        df.to_parquet(path, compression='snappy', index=False)

def clean_old_filled_files():
    files = os.listdir(data_dir)
    for f in files:
        if f.endswith('_filled.csv'):
            os.remove(os.path.join(data_dir, f))

if __name__ == '__main__':
    # clean_old_filled_files()
    start = pd.Timestamp.now()
    main()
    print('Done:', pd.Timestamp.now() - start)