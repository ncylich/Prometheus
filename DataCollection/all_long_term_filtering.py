import os
import gc
import csv
import sys
import tempfile
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


'''
Redundant Tickers:
	ES (E-Mini S&P 500 Futures) - MES (Micro E-mini S&P 500 Futures)
    NQ (E-mini Nasdaq-100 Futures) - MNQ (Micro E-mini Nasdaq-100 Futures)
    RTY (e-Mini Russell 2000) - M2K (Micro E-mini Russell 2000 Index Futures)
    GC (Gold Futures) - MGC (Micro Gold Futures)
    SI (Silver Futures) - SIL (Micro Silver Futures)
    CL (Crude Oil WTI Futures) - MCL (Micro WTI Crude Oil Futures)
    NG (Henry Hub Natural Gas Futures) - QG (Natural Gas Mini Futures)
    YM (Dow Futures Mini) - YM (Dow Futures Mini) (Listed twice as Mini)
    E6 (Euro FX Futures) - E7 (E-mini Euro FX)
    J1 (Japanese Yen Futures) - J7 (E-mini Japanese Yen)
    US (30 Year US Treasury Bond Future) - UB (Ultra US Treasury Bond Futures) (Not exactly a mini but related)
    ZN (10-Year Treasury Note Futures) - TN (Ultra 10-Year US Treasury Note Futures) (Not exactly a mini but related)
    ZF (5-Year Treasury Note Futures) - ZT (2-Year Treasury Note Futures) (Not exactly a mini but related)
	
VERY Related Tickers:
    1. NKD (Nikkei 225 denominated in USD) vs. NIY (Nikkei 225 denominated in JPY)
'''


data_path = '../Local_Data/futures_full_30min_contin_UNadj_11assu1'
dest_path = '../Local_Data/focused_futures_30min'

if not os.path.exists(dest_path):
    os.makedirs(dest_path)


def get_csvs(path=data_path):
    return [f for f in os.listdir(path) if f.endswith('.csv') and not f.endswith('_filled.csv')]


def main():
    csvs = get_csvs()
    cl_index, cl_file = [(i, f) for i, f in enumerate(csvs) if f.startswith("CL_")][0]
    base_df = pd.read_csv(os.path.join(data_path, cl_file))
    base_df = base_df[['date']]
    base_df = base_df.tail(2000)

    base_df['date'] = pd.to_datetime(base_df['date'], utc=True)
    base_df['date'] = base_df['date'].dt.tz_convert('America/New_York')

    print("Loading Files")
    csvs = csvs[:cl_index] + csvs[cl_index + 1:]
    tickers = {}
    for file in tqdm(csvs):
        df = pd.read_csv(os.path.join(data_path, file)).tail(3000)[['date']]
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df['date'] = df['date'].dt.tz_convert('America/New_York')
        tickers[file] = df

    print("Finding best overlaps (greedy)")
    files_used = [cl_file]
    while len(base_df) >= 1700:
        mx = -float('inf')
        mx_file = None
        for file, df in tickers.items():
            new_df = pd.merge(base_df, df, on=['date'], how='inner')
            if len(new_df) > mx:
                mx = len(new_df)
                mx_file = file
                if mx == 2000:
                    break
        base_df = pd.merge(base_df, tickers[mx_file], on=['date'], how='inner')
        files_used.append(mx_file)
        del tickers[mx_file]
        print(f"{mx_file} added -> {len(base_df)} rows")

    files_used = files_used[:-1]
    print(files_used)
    print(len(files_used), "tickers")

    os.system(f"rm -rf {dest_path}/*")  # clear the destination directory
    for file in files_used:
        path = os.path.join(data_path, file)
        os.system(f'cp {path} {dest_path}')  # copy the file to the destination directory

    #     else:
    #         del base_df
    #         base_df = new_df
    # print(len(base_df))

    # Create and display a histogram for the overlaps variable
    # plt.hist(overlaps, bins=20, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of Overlaps')
    # plt.xlabel('Number of Overlaps')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    #
    # print(len([x for x in overlaps if x >= 2000]))


if __name__ == '__main__':
    main()

