import pandas as pd
import pyarrow.parquet as pq
import pandasql as psql


# Define the input and output files
input_file = "aug16-2024-2yrs.parquet"

# Read the parquet file into a DataFrame
print("Reading parquet file...")
df = pq.read_table(input_file).to_pandas()

# Show the top 5 lines of the DataFrame
print(df.columns)
print("Showing top 5 lines of the DataFrame:")
print(df.head(5)['close'])
