import pandas as pd
from security_relationship_analysis import multivariate_regression, plot_2d_graph


def main():
    prop = .1
    degree = 2
    # use_multiplicative_velocities = True

    df = pd.read_parquet(f'../Local_Data/unique_focused_futures_30min/all_long_term_combo.parquet')
    # df = pd.read_parquet(f'../Local_Data/30min_long_term_merged_UNadjusted.parquet')
    tickers = [col for col in df.columns if col.endswith('_close')]
    df = df.tail(int(len(df) * prop))
    df = df[tickers]  # remove all columns that are not tickers
    df = df.rename(columns={col: col.split('_')[0] for col in df.columns})

    # if use_multiplicative_velocities:
    #     for col in df.columns:
    #         df[col] = df[col].pct_change().fillna(0)

    multivar_reg = pd.DataFrame({col: [multivariate_regression(df, col, degree=degree)] for col in df.columns})
    plot_2d_graph(multivar_reg, 'Starting Multivariate R-Squared Values')
    print('Multivariate R-Squared Values (each col with respect to ALL of the others)')
    print(multivar_reg)
    print('X' * 100, '\n')

    start_len = len(df.columns)
    cols = set(df.columns)
    for i in range(start_len // 2):
        multivar_reg = {col: multivariate_regression(df, col, degree=degree) for col in cols}
        mn_col = min(multivar_reg, key=multivar_reg.get)
        print(f"{mn_col}: {multivar_reg[mn_col]}")
        cols.remove(mn_col)

    print("Remaining columns:", cols)
    multivar_reg = pd.DataFrame({col: [multivariate_regression(df, col, degree=degree)] for col in cols})
    plot_2d_graph(multivar_reg, 'Final Multivariate R-Squared Values')
    print('Multivariate R-Squared Values (each col with respect to ALL of the others)')
    print(multivar_reg)
    print('X' * 100, '\n')


if __name__ == "__main__":
    main()