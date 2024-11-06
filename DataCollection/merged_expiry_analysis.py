import pandas as pd

df = pd.read_csv('20241030_merged.csv')

def expiry_date(row):
    expiry = row['expiry']


