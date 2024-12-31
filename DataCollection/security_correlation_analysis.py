import pandas as pd

interval = 1
prop = .1

df = pd.read_parquet(f'../Local_Data/{interval}min_long_term_merged_UNadjusted.parquet')
tickers = [col for col in df.columns if col.endswith('_close')]
df = df.tail(int(len(df) * prop))
df = df[tickers]  # remove all columns that are not tickers

velocity_df = pd.DataFrame()
for ticker in tickers:
    velocity_df[ticker.split('_')[0]] = df[ticker][1:].values / df[ticker][:-1].values

# Calculate the correlation matrix
corr_matrix = velocity_df.corr()
print(corr_matrix)