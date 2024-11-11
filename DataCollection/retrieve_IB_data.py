import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta  # pip install python-dateutil
from concurrent.futures import ThreadPoolExecutor
from calendar import monthrange
import os
import sys
from ib_insync import *
import glob
import pandas as pd

# Base, example, not used
ticker = 'CL'
exchange = 'NYMEX'

today_dt = datetime.today()
# Input date in yyyymmdd format
curMonth = today_dt.strftime("%Y%m")
ib = IB()

cDir = f"./IB_Raw_Data/output_{ticker}/"


def fetch_historical_data(month):
    # get last 4 months data for each month
    createFolder(cDir)
    # createFolder(cDir+"./output/cntrMonth=" + month+"")
    month_obj = datetime.strptime(month, "%Y%m")
    data_months = []
    for i in range(1, 5):
        data_months.append((month_obj - relativedelta(months=+i)).strftime("%Y%m"))
    data_months.reverse()

    for dataMonth in data_months:
        data_month_path = f"{cDir}-cntrMonth={month}-day="

        # createFolder(dataMonthPath)
        for day in get_days_data(dataMonth):
            data_day = datetime.strptime(day, "%Y%m%d")

            if data_day.weekday() >= 5:  # 5 and 6 correspond to Saturday and Sunday respectively
                continue

            if data_day < today_dt - relativedelta(days=+1):
                downloadData(data_month_path, month, dataMonth, day)


def createFolder(path):
    os.makedirs(path, exist_ok=True)


def downloadData(path, month, dataMonth, day):
    # Might have to include "currency='USD'" kwarg
    contract = Future(symbol=ticker, lastTradeDateOrContractMonth=month, exchange=exchange, includeExpired=True)  # Ommit "WOO" at end

    # dt = YYYYMMDD{SPACE}hh:mm:ss[{SPACE}TMZ]
    # startdt = dt.datetime(2023, 1, 15)

    # comtract:         a qualified contract not just a ticker
    # endDateTime:      the start of the loop as it starts with current and goes back through time.
    #                   so it will get everything BEFORE that time.
    #                   Example: edt = '20221230 23:59:59' will get all bars befor the end of 2022
    #                            edt = '' will get everything before the current bar depending on durationStr
    # duration:         the time frame reqHistoricalData will go back
    # NOTE:             duration format is integer{SPACE}unit (S|D|W|M|Y)
    # barSizeSetting:   size of each bar data to return, 1 min, 5 min, 1 hour, 1 day etc
    # whatToShow:       TRADES, MIDPOINT, BID, ASK etc
    # useRTH            True or False, show Regular Trading Hours
    # formatDate:       set to 1, but not sure what it does...

    # edt = '20000101 00:00:01'
    edt = day + ' 23:00:00'

    barsList = []
    # Set the market data type to live data
    ib.reqMarketDataType(2)

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=edt,
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1)

        barsList.append(bars)

        # save to CSV file
        allBars = [b for bars in reversed(barsList) for b in bars]
        df = util.df(allBars)
        df['expiry'] = month
        df['dataMonth'] = dataMonth
        df.to_csv(path + day + '.csv', index=False)
    except Exception as e:
        print("error in fetching data for " + day)
        print(e)


def get_days_data(month):
    # get all the days in a month
    month_obj = datetime.strptime(month, "%Y%m")
    days = []
    for i in range(1, monthrange(month_obj.year, month_obj.month)[1] + 1):
        days.append(datetime(month_obj.year, month_obj.month, i).strftime("%Y%m%d"))
    return days


def get_months(month, num_of_hist_months, num_of_future_months):
    """
    Get list of months

    :param month: input date in yyyymm format
    :type month: str
    :param num_of_hist_months: number of months to go back
    :type num_of_hist_months: int
    :param num_of_future_months: number of months ahead
    :type num_of_future_months: int
    :return: list of months
    :rtype: list[str]
    """
    month_obj = datetime.strptime(month, "%Y%m")
    months = []

    month_obj = month_obj + relativedelta(months=num_of_future_months)
    months.append(month_obj.strftime("%Y%m"))

    # Subtract months from now to past
    for i in range(1, num_of_hist_months + num_of_future_months):
        month_obj = month_obj - relativedelta(months=1)
        months.append(month_obj.strftime("%Y%m"))

    return months


def reorder_cols(df, target_order):
    cols = df.columns.tolist()


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
    df = df.sort_values(['date', 'expiry'])
    df = df.drop_duplicates(subset=['date', 'expiry'], keep='first')
    df = df[['date', 'expiry', 'open', 'high', 'low', 'close', 'volume', 'barCount', 'average']]  # reordering + removed dataMonth

    df.to_parquet(output_file, compression='snappy', index=True)

def main(month=curMonth, num_of_hist_months=38, num_of_future_months=2):
    """
    Main function which takes optional parameters as month in yyyymm format,
    number of months in history, number of months ahead.
    """
    months = get_months(month, num_of_hist_months, num_of_future_months)

    for month in months:
        fetch_historical_data(month)

    print("Data downloaded successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Historical Downloader")
    parser.add_argument("startMonth", type=str, default=curMonth, help="Input current date in yyyymm format.",
                        nargs='?')
    parser.add_argument("monthsBackInTime", type=int, default=38, help="How many months to go back.", nargs='?')
    parser.add_argument("monthsAhead", type=int, default=4, help="How many months to go ahead.", nargs='?')

    start = datetime.now()

    ib.connect('127.0.0.1', 7497, clientId=0)

    args = parser.parse_args()

    # GC - Gold, CL - Crude Oil, NG - Natural Gas, ES - E-mini S&P 500, ZN - 10-Year T-Note, DX - US Dollar Index, HG - Copper, SI - Silver
    stock_list = [['GC', 'COMEX'],
                  ['CL', 'NYMEX'],
                  ['NG', 'NYMEX'],
                  ['ES', 'CME'],
                  ['ZN', 'CBOT'],
                  ['DX', 'NYBOT'],
                  ['HG', 'COMEX'],]
                  # ['SI', 'COMEX']]  # SILVER not working

    for stock in stock_list:
        curr_start = datetime.now()

        ticker, exchange = stock
        today = today_dt.strftime('%Y%m%d')
        cDir = f"./IB_Raw_Data/{today}_output_{ticker}/"
        parquet = f"./IB_Processed_Data/{today}_output_{ticker}.parquet"

        main(args.startMonth, args.monthsBackInTime, args.monthsAhead)
        convert_csv_to_parquet(cDir, parquet)

        time_taken = datetime.now() - curr_start
        # print formatted time: mm:ss
        print(f"Time taken for {ticker}: {time_taken.seconds // 60}:{time_taken.seconds % 60:02d}")

    ib.disconnect()

    time_taken = datetime.now() - start
    # print formatted time: mm:ss
    print(f"Total time taken: {time_taken.seconds // 60}:{time_taken.seconds % 60:02d}")
