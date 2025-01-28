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
    train_df = df.head(int(len(df) * prop))

    multivar_reg_train = {col: multivariate_regression(train_df, col, degree=degree) for col in train_df.columns}
    multivar_reg = pd.DataFrame({col: [score] for col, (score, _) in multivar_reg_train.items()})[tickers]

    plot_2d_graph(multivar_reg, 'Starting Multivariate R-Squared Values')
    print('Multivariate R-Squared Values (each col with respect to ALL of the others)')
    print(multivar_reg)
    print('X' * 100, '\n')

    mutlivar_test_reg = pd.DataFrame({col: [test_multivariate_regression(multivar_reg_train[col][1], test_df, col, degree)]
                                      for col in train_df.columns})[tickers]
    plot_2d_graph(mutlivar_test_reg, 'Starting Test Multivariate R-Squared Values')

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
    plot_2d_graph(multivar_reg, 'Final Multivariate R-Squared Values')
    print('Multivariate R-Squared Values (each col with respect to ALL of the others)')
    print(multivar_reg)
    print('X' * 100, '\n')

    mutlivar_test_reg = pd.DataFrame({col: [test_multivariate_regression(multivar_reg_train[col][1], test_df, col, degree)]
                                      for col in cols})[remaining_tickers]
    plot_2d_graph(mutlivar_test_reg, 'Final Test Multivariate R-Squared Values')


if __name__ == "__main__":
    main()