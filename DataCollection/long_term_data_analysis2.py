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

path = os.path.join('..', 'Local_Data', f'1min_long_term_merged_UNadjusted_filled.parquet')
data = pd.read_parquet(path)
# formatting date
data['date'] = pd.to_datetime(data['date'], utc=True)
data['date'] = data['date'].dt.tz_convert('America/New_York')

start_prop = 0.99
start_idx = int(len(data) * start_prop)

start_date = data.iloc[start_idx]['date']
prev_date = start_date
curr_day_of_year = start_date.day_of_year

time_spans = []

def get_minutes(date):
    # only get the minutes and hours
    return date.minute / 60 + date.hour

for i in tqdm(range(start_idx + 1, len(data))):
    date = data.iloc[i]['date']
    day = date.day_of_year

    if day != curr_day_of_year:
        time_spans.append((start_date, get_minutes(start_date), get_minutes(prev_date)))

        start_date = date
        prev_date = date
        curr_day_of_year = day
    else:  # Only adding to gap if it's the same day and not the first minute of the day
        prev_date = date

time_spans.append((start_date, get_minutes(start_date), get_minutes(prev_date)))


time_spans_df = pd.DataFrame(time_spans, columns=['start date', 'start time', 'end time'])

days = (4, 5, 6)  # Friday, Saturday, Sunday
time_spans_df = time_spans_df[~time_spans_df['start date'].dt.day_of_week.isin(days)]


# BASiC GRAPHS of START AND END TIMES
time_spans_df.plot(x='start date', y='start time')
plt.title('Start Time Over Time')
plt.show()

time_spans_df.plot(x='start date', y='end time')
plt.title('End Time Over Time')
plt.show()

# days of week, starts with monday = 0
start = time_spans_df.iloc[0]['start date']
end = time_spans_df.iloc[-1]['start date']
days = pd.date_range(
    start=start,
    end=end,
    freq='D'
)

# make df of days and whether theu're in time_spans_df
days_df = pd.DataFrame(days, columns=['date'])
# format as year-month-day
days_df['day'] = days_df['date'].dt.strftime('%Y-%m-%d')
time_spans_df['day'] = time_spans_df['start date'].dt.strftime('%Y-%m-%d')

time_span_days = set(time_spans_df['day'].unique())
days_df['in_time_spans'] = days_df['day'].isin(time_span_days)
days_df['in_time_spans'] = days_df['in_time_spans'].astype(int)

# plot
days_df.plot(x='date', y='in_time_spans')
plt.title('Days in Time Spans')
plt.show()

time_spans_df['day of week'] = time_spans_df['start date'].dt.day_of_week
day_to_freq = {day: len(time_spans_df[time_spans_df['day of week'] == day]) for day in range(7)}
day_to_freq = pd.DataFrame(day_to_freq.items(), columns=['day of week', 'freq'])
idx_to_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_to_freq['day of week'] = day_to_freq['day of week'].map(lambda x: idx_to_day[x])

day_to_freq.plot(x='day of week', y='freq', kind='bar')
plt.title('Day of Week Frequency')
plt.show()
