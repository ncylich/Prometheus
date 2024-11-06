from merged_expiry_analysis import approx_expiry_dist
import pandas as pd


DATE = '20241030'

def squeeze_df(df):
    df['expiry-dist'] = df.apply(approx_expiry_dist, axis=1)

    tickers = set([col.split('_')[0] for col in df.columns if "_" in col])  #

    squeezed_df = pd.DataFrame()
    for i in range(1, 5):  # between 1 and 4, inclusive
        offsets = {ticker: 0 for ticker in tickers}
        curr_df = df[df["expiry-dist"] == i].copy()

        for j in range(1, len(curr_df)):
            for ticker in tickers:
                open_col = f"{ticker}_open"
                close_col = f"{ticker}_close"

                if curr_df.iloc[j]['date'].date() != curr_df.iloc[j-1]['date'].date():
                    offsets[ticker] = curr_df.iloc[j - 1][close_col] - curr_df.iloc[j][open_col]

                curr_df.loc[curr_df.index[j], open_col] += offsets[ticker]
                curr_df.loc[curr_df.index[j], close_col] += offsets[ticker]

        if len(squeezed_df):
            squeezed_df = pd.concat([squeezed_df, curr_df])
        else:
            squeezed_df = curr_df

    squeezed_df = squeezed_df.sort_values(by=['date', 'expiry'])

    squeezed_df.reset_index(inplace=True)
    squeezed_df.drop(columns=['index', 'Unnamed: 0'], inplace=True)
    return squeezed_df

if __name__ == '__main__':
    start_df = pd.read_csv(f'{DATE}_merged.csv')
    start_df['date'] = pd.to_datetime(start_df['date'], utc=True)
    start_df['date'] = start_df['date'].dt.tz_convert('America/New_York')
    final_df = squeeze_df(start_df)
    final_df.to_csv(f"{DATE}_merged_squeezed.csv", index=True)
