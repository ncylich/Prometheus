import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

'''
ANALYZES THE START AND END TIMES OF THE DATA
Result: Follows a weekly cycle/pattern:
    No Results on Saturday
    Sunday starts later then rest and fewer results
    Friday ends later
'''

path = os.path.join('..', 'Local_Data', f'1min_long_term_merged_UNadjusted.parquet')
data = pd.read_parquet(path)
# formatting date
data['date'] = pd.to_datetime(data['date'], utc=True)
data['date'] = data['date'].dt.tz_convert('America/New_York')

# Start of attempt to accelerate with vectorization
remove_cols = set(data.columns) - {'date'}
data = data.drop(columns=list(remove_cols))
data['day'] = data['date'].dt.strftime('%Y-%m-%d')

data = data.groupby('day').agg({'date': ['min', 'max']})
# expanding date min and max into normal cols
data.columns = data.columns.map('_'.join)
data = data.reset_index()
data = data.rename(columns={'date_min': 'start', 'date_max': 'end'})

def hour_time(date):
    return date.hour + date.minute / 60
data['start_time'] = data['start'].apply(hour_time)
data['end_time'] = data['end'].apply(hour_time)
data['day_of_week'] = data['start'].dt.day_of_week


# BASiC GRAPHS of START AND END TIMES
data.plot(x='start', y='start_time')
plt.title('Start Time Over Time')
plt.show()

data.plot(x='end', y='end_time')
plt.title('End Time Over Time')
plt.show()

# aggregating to count freq of each day of week
day_freq = data.groupby('day_of_week').agg({'day': 'count'})
day_freq = day_freq.rename(columns={'day': 'freq'})
day_freq = day_freq.reset_index()

idx_to_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_freq['day_of_week_str'] = day_freq['day_of_week'].map(lambda x: idx_to_day[x])

day_freq.plot(x='day_of_week_str', y='freq', kind='bar')
plt.title('Day of Week Frequency')
plt.subplots_adjust(bottom=0.3)  # Giving space for x-axis labels
plt.show()
