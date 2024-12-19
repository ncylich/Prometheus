import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os


path = os.path.join('..', 'Local_Data', f'1min_long_term_merged_UNadjusted.parquet')
data = pd.read_parquet(path)
# formatting date
data['date'] = pd.to_datetime(data['date'], utc=True)
data['date'] = data['date'].dt.tz_convert('America/New_York')

last_date = data.iloc[0]['date']
last_day_of_year = last_date.day_of_year
last_minute = last_date.minute + last_date.hour * 60
first_minute = last_minute
gap_span = 0
gap_num = 0

day_to_gap = []

for i in tqdm(range(1, len(data))):
    date = data.iloc[i]['date']
    day = date.day_of_year
    total_minutes = date.minute + date.hour * 60

    if day != last_day_of_year:
        day_to_gap.append((last_date, gap_span, gap_num, last_minute - first_minute))

        last_date = date
        last_day_of_year = day
        first_minute = total_minutes
        gap_span = 0
        gap_num = 0
    else:  # Only adding to gap if it's the same day and not the first minute of the day
        diff = total_minutes - last_minute - 1  # -1 for expected differences
        if diff > 0:
            gap_span += diff
            gap_num += 1

    last_minute = total_minutes  # updating last minute every iteration

day_to_gap.append((last_date, gap_span, gap_num, last_minute - first_minute))

# loading date to gap as a dataframe
date_to_gap_df = pd.DataFrame(day_to_gap, columns=['date', 'gap span', 'gap num', 'total minutes'])
date_to_gap_df['proportion skipped'] = date_to_gap_df['gap span'] / date_to_gap_df['total minutes']

# BASIC GRAPHS OF DAILY GAP ANALYSIS
# date_to_gap_df.plot(x='date', y='gap span')
# plt.title('Span of Gaps Over Time (Daily)')
# plt.show()
#
# # graphing over time for gap num
# date_to_gap_df.plot(x='date', y='gap num')
# plt.title('Number of Gaps Over Time (Daily)')
# plt.show()

# JOINTLY graphing over time for gap span and gap num
# fig, ax1 = plt.subplots()
#
# color = 'tab:blue'
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Span of Gaps', color=color)
# ax1.plot(date_to_gap_df['date'], date_to_gap_df['gap span'], color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:red'
# ax2.set_ylabel('Number of Gaps', color=color)  # we already handled the x-label with ax1
# ax2.plot(date_to_gap_df['date'], date_to_gap_df['gap num'], color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title('Gap Span and Number of Gaps Over Time')
# plt.show()


# Aggregating into gaps per month
month_to_gap = []
last_date = day_to_gap[0][0]
gap_span = day_to_gap[0][1]
gap_num = day_to_gap[0][2]
total_minutes = day_to_gap[0][3]

for i in range(1, len(day_to_gap)):
    date, span, num, minutes = day_to_gap[i]
    month = date.month

    if month != last_date.month:
        month_to_gap.append((last_date, gap_span, gap_num, total_minutes))

        last_date = date
        gap_span = 0
        gap_num = 0
        total_minutes = 0

    # Adding gaps from each day of month (including the first)
    gap_span += span
    gap_num += num
    total_minutes += minutes

month_to_gap.append((last_date, gap_span, gap_num, total_minutes))

# loading month to gap as a dataframe
month_to_gap_df = pd.DataFrame(month_to_gap, columns=['month', 'gap span', 'gap num', 'total minutes'])
month_to_gap_df['proportion skipped'] = month_to_gap_df['gap span'] / month_to_gap_df['total minutes']


# BASIC GRAPHS OF MONTHLY GAP ANALYSIS
# # graphing over time for gap span
# month_to_gap_df.plot(x='month', y='gap span')
# plt.title('Span of Gaps Over Time (Monthly)')
# plt.show()
#
# # graphing over time for gap num
# month_to_gap_df.plot(x='month', y='gap num')
# plt.title('Number of Gaps Over Time (Monthly)')
# plt.show()

# COMBINED GRAPH OF MONTHLY AND DAILY GAP ANALYSIS
# Set number as blue and span as red
# Set daily as lighter and monthly as darker
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# Plotting daily gap num
date_to_gap_df.plot(x='date', y='gap num', ax=axs[0, 0])
axs[0, 0].set_title('Number of Gaps Over Time (Daily)')
axs[0, 0].get_lines()[0].set_color('lightblue')

# Plotting monthly gap num
month_to_gap_df.plot(x='month', y='gap num', ax=axs[0, 1])
axs[0, 1].set_title('Number of Gaps Over Time (Monthly)')
axs[0, 1].get_lines()[0].set_color('blue')

# Plotting daily gap span
date_to_gap_df.plot(x='date', y='gap span', ax=axs[1, 0])
axs[1, 0].set_title('Span of Gaps Over Time (Daily)')
axs[1, 0].get_lines()[0].set_color('lightcoral')

# Plotting monthly gap span
month_to_gap_df.plot(x='month', y='gap span', ax=axs[1, 1])
axs[1, 1].set_title('Span of Gaps Over Time (Monthly)')
axs[1, 1].get_lines()[0].set_color('red')

# Plotting daily proportion of minutes skipped
date_to_gap_df.plot(x='date', y='proportion skipped', ax=axs[2, 0])
axs[2, 0].set_title('Proportion of Minutes Skipped (Daily)')
axs[2, 0].get_lines()[0].set_color('purple')

# Plotting monthly proportion of minutes skipped
month_to_gap_df.plot(x='month', y='proportion skipped', ax=axs[2, 1])
axs[2, 1].set_title('Proportion of Minutes Skipped (Monthly)')
axs[2, 1].get_lines()[0].set_color('darkviolet')

plt.tight_layout()
plt.show()
