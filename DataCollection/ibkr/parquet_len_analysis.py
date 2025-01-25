from pyarrow.parquet import read_table

DATE = "20241111"

files = f"""{DATE}_output_CL.parquet
# {DATE}_output_DX.parquet  # Decreases usable data periods ~80%
{DATE}_output_ES.parquet
{DATE}_output_GC.parquet  # Decreases ~20%
{DATE}_output_HG.parquet
{DATE}_output_NG.parquet
{DATE}_output_ZN.parquet
"""

files = [f'IB_Processed_Data/{file}' for file in files.split("\n") if file and file[0] != '#']

date_expiry_overlap = None
date_overlap = None
for file in files:
    file = file.split('#')[0].strip()
    print(f"Reading {file}")
    table = read_table(file).to_pandas()
    date_expirys = set()
    dates = set()
    data = table[['date', 'expiry']].drop_duplicates(['date', 'expiry'])
    for date, expiry in data.values:
        date_expirys.add((date, expiry))
        dates.add(date)
    print(f'Time Periods: {len(dates)}, Time Expiry Combinations: {len(date_expirys)}\n')

    date_expiry_overlap = date_expirys if date_expiry_overlap is None else date_expiry_overlap.intersection(date_expirys)
    date_overlap = dates if date_overlap is None else date_overlap.intersection(dates)

days = set([date.split(' ')[0] for date, _ in date_expiry_overlap])
print(f'Days in common: {len(days)}')

print(f'Time Periods in common: {len(date_overlap)}')
print(f'Time Expiry Combinations in common: {len(date_expiry_overlap)}')

dates_from_date_expiry_overlap = set([date for date, _ in date_expiry_overlap])
print(f'Time Periods from Time Expiry Combinations: {len(dates_from_date_expiry_overlap)}')

print()
date_expiry_dist_overlap = []
for date, expiry in date_expiry_overlap:
    expiry_year, expiry_month = expiry // 100, expiry % 100
    date = date.split('-')
    date_year, date_month = int(date[0]), int(date[1])
    dist = (expiry_year * 12 + expiry_month) - (date_year * 12 + date_month)
    date_expiry_dist_overlap.append((date, dist))
for i in range(1, 5):
    dates = [date for date, dist in date_expiry_dist_overlap if dist == i]
    print(f'Expiry Distance {i} Months: {len(dates)}')
