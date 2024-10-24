import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import monthrange
import os
import asyncio
from ib_async import IB, util
from ib_async.contract import Future
import nest_asyncio

ticker = 'HH'
today_dt = datetime.today()
curMonth = today_dt.strftime("%Y%m")
cDir = f"./output_{ticker}/"

async def fetch_historical_data(ib, month):
    createFolder(cDir)
    month_obj = datetime.strptime(month, "%Y%m")
    data_months = [(month_obj - relativedelta(months=i)).strftime("%Y%m") for i in range(1, 5)]
    data_months.reverse()

    for dataMonth in data_months:
        data_month_path = f"{cDir}-cntrMonth={month}-day="
        for day in get_days_data(dataMonth):
            data_day = datetime.strptime(day, "%Y%m%d")
            if data_day.weekday() >= 5 or data_day >= today_dt:
                continue
            await downloadData(ib, data_month_path, month, dataMonth, day)

def createFolder(path):
    os.makedirs(path, exist_ok=True)

async def downloadData(ib, path, month, dataMonth, day):
    contract = Future(symbol=ticker, lastTradeDateOrContractMonth=month, exchange='NYMEX', includeExpired=True)
    edt = day + ' 23:00:00'

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=edt,
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        df = util.df(bars)
        df['expiry'] = month
        df['dataMonth'] = dataMonth
        df.to_csv(path + day + '.csv', index=False)
    except Exception as e:
        print(f"Error fetching data for {day}: {e}")

def get_days_data(month):
    month_obj = datetime.strptime(month, "%Y%m")
    return [datetime(month_obj.year, month_obj.month, i).strftime("%Y%m%d") for i in range(1, monthrange(month_obj.year, month_obj.month)[1] + 1)]

async def main(month=curMonth, num_of_hist_months=38, num_of_future_months=2):
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=0)
    months = [(datetime.strptime(month, "%Y%m") + relativedelta(months=i)).strftime("%Y%m") for i in range(-num_of_hist_months, num_of_future_months + 1)]

    tasks = [fetch_historical_data(ib, m) for m in months]
    await asyncio.gather(*tasks)

    await ib.disconnect()
    print("Data downloaded successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Historical Downloader")
    parser.add_argument("startMonth", type=str, default=curMonth, help="Input current date in yyyymm format.", nargs='?')
    parser.add_argument("monthsBackInTime", type=int, default=38, help="How many months to go back.", nargs='?')
    parser.add_argument("monthsAhead", type=int, default=4, help="How many months to go ahead.", nargs='?')
    args = parser.parse_args()

    start = datetime.now()

    nest_asyncio.apply()  # Allows nested asyncio loops

    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If the event loop is already running, use it
        task = loop.create_task(main(args.startMonth, args.monthsBackInTime, args.monthsAhead))
        loop.run_until_complete(task)
    else:
        asyncio.run(main(args.startMonth, args.monthsBackInTime, args.monthsAhead))

    time_taken = datetime.now() - start
    # print formatted time: mm:ss
    print(f"Time taken: {time_taken.seconds//60}:{time_taken.seconds%60:02d}")
