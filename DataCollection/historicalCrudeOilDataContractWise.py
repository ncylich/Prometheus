import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta # pip install python-dateutil
from calendar import monthrange
import os
from ib_insync import *

todayDt = datetime.today()
# Input date in yyyymmdd format
curMonth = todayDt.strftime("%Y%m")
ib = IB()

cDir = "./output/"


def fetch_historical_data(month):
    #get last 4 months data for each month
    createFolder(cDir)
    #createFolder(cDir+"./output/cntrMonth=" + month+"")
    month_obj = datetime.strptime(month, "%Y%m")
    dataMonths = []
    for i in range(1,5):
        dataMonths.append((month_obj - relativedelta(months=+i)).strftime("%Y%m"))
    dataMonths.reverse()  

    for dataMonth in dataMonths:
        dataMonthPath = cDir+"cntrMonth=" + month+"-day="
        #createFolder(dataMonthPath)
        for day in getDaysData(dataMonth):
            dataDay = datetime.strptime(day, "%Y%m%d")

            if dataDay.weekday() >= 5: # 5 and 6 correspond to Saturday and Sunday respectively
                continue


            if (dataDay < todayDt - relativedelta(days=+1)):
                downloadData(dataMonthPath, month, dataMonth, day)


def createFolder(path):
    os.makedirs(path,exist_ok=True)

def downloadData(path, month, dataMonth, day):
    

    contract = Future(symbol='CL', lastTradeDateOrContractMonth=month, exchange='NYMEX', includeExpired=True )

    #dt = YYYYMMDD{SPACE}hh:mm:ss[{SPACE}TMZ]
    #startdt = dt.datetime(2023, 1, 15)


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


    #edt = '20000101 00:00:01'
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


    

def getDaysData(month):
    # get all the days in a month
    month_obj = datetime.strptime(month, "%Y%m")
    days = []
    for i in range(1, monthrange(month_obj.year, month_obj.month)[1] + 1):
        days.append(datetime(month_obj.year, month_obj.month, i).strftime("%Y%m%d"))
    return days

        

def main(month=curMonth, num_of_hist_months=38, num_of_future_months=2):
    """
    Main function which takes optional parameters as month in yyyymm format,
    number of months in history, number of months ahead.
    """
    months = get_months(month, num_of_hist_months, num_of_future_months)

    
    for month in months:
        fetch_historical_data(month)


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




if __name__ == "__main__":
    import sys
    #main(*sys.argv[1:])
    parser = argparse.ArgumentParser(description="Batch Historical Downloader")
    parser.add_argument("startMonth", type=str, default=curMonth, help="Input current date in yyyymm format.")
    parser.add_argument("monthsBackInTime", type=int, default=38, help="How many months to go back.")
    parser.add_argument("monthsAhead", type=int, default=2, help="How many months to go ahead.")

   

    ib.connect('127.0.0.1', 7497, clientId=0)

    args = parser.parse_args()
    main(args.startMonth, args.monthsBackInTime, args.monthsAhead)

    ib.disconnect()









