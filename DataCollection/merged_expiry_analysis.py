import pandas as pd

# Approximate expiry distance (in months) from date
def approx_expiry_dist(row):
    expiry = row['expiry']
    year, month = expiry // 100, expiry % 100
    absolute_expiry = year * 12 + month

    date = row['date']
    year, month = date.year, date.month
    absolute_date = year * 12 + month

    return absolute_expiry - absolute_date

if __name__ == "__main__":
    df = pd.read_csv('20241030_merged.csv')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['expiry_dist'] = df.apply(approx_expiry_dist, axis=1)
    for i in range(1, 5):  # between 1 and 4, inclusive
        print(f'num with expiry dist = {i}: {len(df[df["expiry_dist"] == i])}')

'''
num with expiry dist = 1: 13500
num with expiry dist = 2: 20250
num with expiry dist = 3: 20790
num with expiry dist = 4: 18900
'''
