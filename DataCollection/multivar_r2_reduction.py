import numpy as np
import pandas as pd
from security_relationship_analysis import multivariate_regression, plot_2d_graph, r_squared
import os


def test_multivariate_regression(model, test_df, target_col, degree):
    other_cols = [col for col in test_df.columns if col != target_col]

    # Prepare X, y for testing in the same way as training
    start_X_test = test_df[other_cols].values
    X_test = start_X_test
    y_test = test_df[target_col].values

    for i in range(2, degree + 1):
        X_test = np.concatenate((X_test, start_X_test ** i), axis=1)

    # test_r2 = model.score(X_test, y_test)
    test_r2 = r_squared(X_test, y_test, model)
    return test_r2


def main():
    prop = .2
    degree = 1  # Surprisingly, the R^2 values are highest when degree 1

    df = pd.read_parquet(f'../Local_Data/unique_focused_futures_30min/interpolated_all_long_term_combo.parquet')
    # df = pd.read_parquet(f'../Local_Data/30min_long_term_merged_UNadjusted.parquet')
    tickers = [col for col in df.columns if col.endswith('_close')]
    df = df.tail(int(len(df) * prop))
    df = df[tickers]  # remove all columns that are not tickers
    df = df.rename(columns={col: col.split('_')[0] for col in df.columns})
    tickers = [col.split('_')[0] for col in tickers]
    for col in df.columns:
        df[col] = df[col].pct_change().fillna(0)

    # Shuffle the DataFrame
    np.random.seed(0)
    shuffled_indices = np.random.permutation(len(df))
    df = df.iloc[shuffled_indices]

    # Split the DataFrame into training and testing
    test_df = df.tail(int(len(df) * prop))
    train_df = df.head(len(df) - int(len(df) * prop))

    # Normalization attempts (didn't work)
    # train_df = pd.concat([train_df] * 10, ignore_index=True)
    #
    # # Add noise to train_df to normalize results
    # std_devs = train_df.std()
    # noise_scale = 1e-2
    # for col in train_df.columns:
    #     noise = np.random.normal(loc=0, scale=std_devs[col] * noise_scale, size=len(train_df))
    #     train_df.loc[:, col] = train_df[col] + noise

    multivar_reg_train = {col: multivariate_regression(train_df, col, degree=degree) for col in train_df.columns}
    multivar_reg = pd.DataFrame({col: [score] for col, (score, _) in multivar_reg_train.items()})[tickers]

    plot_2d_graph(multivar_reg, f'Starting Multivariate R-Squared Values, degree={degree}')
    print('Multivariate R-Squared Values (each col with respect to ALL of the others)')
    print(multivar_reg)
    print('X' * 100, '\n')

    mutlivar_test_reg = pd.DataFrame({col: [test_multivariate_regression(multivar_reg_train[col][1], test_df, col, degree)]
                                      for col in train_df.columns})[tickers]
    plot_2d_graph(mutlivar_test_reg, f'Starting Test Multivariate R-Squared Values, degree={degree}')

    start_len = len(train_df.columns)
    cols = set(train_df.columns)
    for i in range(start_len // 2):
        multivar_reg = {col: multivariate_regression(train_df, col, degree=degree)[0] for col in cols}
        mn_col = min(multivar_reg, key=multivar_reg.get)
        print(f"{mn_col}: {multivar_reg[mn_col]}")
        cols.remove(mn_col)

    print("Remaining columns:", cols)
    remaining_tickers = [ticker for ticker in tickers if ticker in cols]
    multivar_reg_train = {col: multivariate_regression(train_df, col, degree=degree) for col in cols}
    multivar_reg = pd.DataFrame({col: [score] for col, (score, _) in multivar_reg_train.items()})[remaining_tickers]
    plot_2d_graph(multivar_reg, f'Final Multivariate R-Squared Values, degree={degree}')
    print('Multivariate R-Squared Values (each col with respect to ALL of the others)')
    print(multivar_reg)
    print('X' * 100, '\n')

    mutlivar_test_reg = pd.DataFrame({col: [test_multivariate_regression(multivar_reg_train[col][1], test_df, col, degree)]
                                      for col in cols})[remaining_tickers]
    plot_2d_graph(mutlivar_test_reg, f'Final Test Multivariate R-Squared Values, degree={degree}')

    # sorting the columns by R^2 value
    sorted_cols = multivar_reg_train.keys()
    sorted_cols = sorted(sorted_cols, key=lambda x: multivar_reg_train[x][0], reverse=True)
    print("Sorted columns by R^2 value")
    print(sorted_cols)


if __name__ == "__main__":
    main()