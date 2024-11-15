import os
import sys
import pandas as pd
import pyarrow as pq


COLS = {
    'date': 'str',
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
}

unadjusted_postfix = '_full_1min_continuous_UNadjusted'
adjusted_postfix = '_full_1min_continuous_ratio_adjusted'
data_dir = 'long_term_data'

def get_files(path=data_dir):
    """
    Get all files in a directory
    """
    all_files = os.listdir(path)
    data_files = [f for f in all_files if unadjusted_postfix in f or adjusted_postfix in f]
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

def load_df(file):
    if file.endswith('.csv'):
        df = pd.read_csv(file, dtype=COLS)
    elif file.endswith('.parquet'):
        df = pd.read_parquet(file)
    else:
        raise ValueError(f'File type not supported for: {file}')

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
    cols = {col: f"{ticker}_{col.strip()}" for col in cols}
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

    dfs = [load_and_format_df(file) for file in data_files]
    merged_df = dfs[0]
    for stock_df in dfs[1:]:
        merged_df = merged_df.merge(stock_df, on=['date'], how='inner')
    return merged_df

def main():
    files = get_files()

    df = merge_data(files)
    print('UNadjusted number of row:', len(df))
    path = os.path.join(data_dir, '1min_long_term_merged_UNadjusted.parquet')
    df.to_parquet(path, compression='snappy', index=True)

    df = merge_data(files, adjusted=True)
    print('adjusted number of row:', len(df))
    path = os.path.join(data_dir, '1min_long_term_merged_adjusted.parquet')
    df.to_parquet(path, compression='snappy', index=True)

if __name__ == '__main__':
    main()
