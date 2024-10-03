import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import gc


def read_parquet(input_file, smush_times=False, expected_expiry_dist=-1, y_var='close'):
    # Read the parquet file into a DataFrame
    print("Reading parquet file...")
    df = pq.read_table(input_file).to_pandas()
    # re-setting up the index
    df = df.reset_index(drop=False)
    df = df.rename(columns={'date': 'ds', y_var: 'y'})

    df['ds'] = pd.to_datetime(df['ds'], utc=True)
    df['ds'] = df['ds'].dt.tz_convert('America/New_York')

    if expected_expiry_dist >= 0:
        def expiry_dist(expiry, ds):
            expiry_month, expiry_year = expiry % 100, expiry // 100
            total_expiry_month = expiry_month + expiry_year * 12
            total_ds_month = ds.month + ds.year * 12
            return total_expiry_month - total_ds_month
        df['expiry_dist'] = df.apply(lambda x: expiry_dist(x['expiry'], x['ds']), axis=1)
        df = df[df['expiry_dist'] == expected_expiry_dist]
        df = df.drop(columns=['expiry_dist'])

    df = df[['ds', 'y']]
    df = df.sort_values(by='ds')


    if smush_times: # doesn't work well because of the jumps
        offset = 0.0
        new_df_values = []
        for i in range(0, len(df) - 1):
            # check if the date has changed and updating offset
            if df.iloc[i]['ds'].date() != df.iloc[i + 1]['ds'].date():
                diff = df.iloc[i]['y'] - df.iloc[i + 1]['y']
                offset += diff
            # not adding detected jump to remove redundancy
            else:
                new_df_values.append(df.iloc[i]['y'] + offset)
        # adding the last value
        last_idx = len(df) - 1
        if df.iloc[last_idx]['ds'].date() == df.iloc[last_idx - 1]['ds'].date():
            new_df_values.append(df.iloc[last_idx]['y'] + offset)

        # remove and clean old df
        del df
        gc.collect()

        # setting up new df
        df = pd.DataFrame({'y': new_df_values})
        df['ds'] = pd.date_range(start='1/1/2020', periods=len(df), freq='5min')  # adding date in 5 minute intervals
    return df


def print_forecasts(input_data, actual, forecasts, forecast_col):
    # print(forecasts)

    # Plot actual values
    plt.plot(input_data['ds'], input_data['y'], label='Input')
    plt.plot(actual['ds'], actual['y'], label='Actual')

    # Plot forecasted values
    plt.plot(forecasts['ds'], forecasts[forecast_col], label='Forecast', linestyle='--')

    plt.legend()
    plt.show()

def test_train_split(df, test_size_ratio):
    test_len = int(len(df) * test_size_ratio)
    return df.head(len(df) - test_len).copy(), df.tail(test_len).copy()


if __name__ == "__main__":
    df_prepared = read_parquet("aug16-2024-2yrs.parquet", smush_times=True, expected_expiry_dist=3)
