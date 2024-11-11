from pyarrow.parquet import read_table

DATE = "20241111"

files = f"""{DATE}_output_CL.parquet
# {DATE}_output_DX.parquet  # Decreases usable data periods ~80%
{DATE}_output_ES.parquet  # Decreases ~20%
{DATE}_output_GC.parquet
{DATE}_output_HG.parquet
{DATE}_output_NG.parquet
{DATE}_output_ZN.parquet
"""

files = [f'IB_Processed_Data/{file}' for file in files.split("\n") if file and file[0] != '#']
data_periods = None
for file in files:
    file = file.split('#')[0].strip()
    print(f"Reading {file}")
    table = read_table(file).to_pandas()
    data_expirys = []
    dates = table[['date', 'expiry']].drop_duplicates(['date', 'expiry'])
    for date, expiry in dates.values:
        data_expirys.append((date, expiry))
    print('Time Periods:', len(data_expirys), '\n')
    data_expirys = set(data_expirys)
    if data_periods is None:
        data_periods = data_expirys
    else:
        data_periods = data_periods.intersection(data_expirys)

print('Time Periods in common:', len(data_periods))
