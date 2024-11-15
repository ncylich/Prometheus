import glob
import os
import pandas as pd
import pyarrow as pa # pip install pyarrow
import pyarrow.parquet as pq # pip install pyarrow


def convert_csv_to_parquet(input_folder, output_file):
    """
    converts all csv files inside a folder into a single parquet with snappy compression

    :param input_folder: folder with csv files
    :param output_file: output parquet file
    """
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_list.append(df)

    df = pd.concat(df_list)

    df.set_index('expiry', inplace=True)
    #df.set_index('dataMonth', inplace=True)

    df.to_parquet(output_file, compression='snappy', index=True)


if __name__ == "__main__":

    print("start")
    input_folder = "./flat-data-in-csv/"
    output_file = "./parquet-aug16-2024/aug16-2024-2yrs.parquet"
    convert_csv_to_parquet(input_folder, output_file)

    print("done")