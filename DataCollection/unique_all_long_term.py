import os
import pandas as pd

'''
Redundant Tickers:
    1. ES (E-Mini S&P 500 Futures) - MES (Micro E-mini S&P 500 Futures)
    2. NQ (E-mini Nasdaq-100 Futures) - MNQ (Micro E-mini Nasdaq-100 Futures)
    3. RTY (e-Mini Russell 2000) - M2K (Micro E-mini Russell 2000 Index Futures)
    4. GC (Gold Futures) - MGC (Micro Gold Futures)
    5. SI (Silver Futures) - SIL (Micro Silver Futures)
    6. NG (Henry Hub Natural Gas Futures) - QG (Natural Gas Mini Futures)
    7. E6 (Euro FX Futures) - E7 (E-mini Euro FX)
    8. ZF (5-Year Treasury Note Futures) - ZT (2-Year Treasury Note Futures) (Not exactly a mini but related)
'''

pairs = [("ES", "MES"), ("NQ", "MNQ"), ("RTY", "M2K"), ("GC", "MGC"), ("SI", "SIL"), ("NG", "QG"), ("E6", "E7"), ("ZF", "ZT")]

data_path = '../Local_Data/focused_futures_30min'
dest_path = '../Local_Data/unique_focused_futures_30min'

files = os.listdir(data_path)
remove_files = set()
for pair in pairs:
    try:
        file1 = [f for f in files if f.startswith(f"{pair[0]}_")][0]
        file2 = [f for f in files if f.startswith(f"{pair[1]}_")][0]
    except Exception as e:
        print(f"Couldn't find both files for pair {pair}")
        continue
    df1 = pd.read_csv(os.path.join(data_path, file1))
    df2 = pd.read_csv(os.path.join(data_path, file2))
    remove_files.add(file1 if len(df1) > len(df2) else file2)

keep_files = [f for f in files if f not in remove_files]
os.system(f"rm -rf {dest_path}/*")  # clear the destination directory
for file in keep_files:
    path = os.path.join(data_path, file)
    os.system(f'cp {path} {dest_path}')  # copy the file to the destination directory
