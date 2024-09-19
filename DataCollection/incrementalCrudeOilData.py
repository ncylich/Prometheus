import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta # pip install python-dateutil
from calendar import monthrange
import os
from ib_insync import *
import pyarrow as pa # pip install pyarrow
import pyarrow.parquet as pq # pip install pyarrow

todayDt = datetime.today()
# Input date in yyyymmdd format
curMonth = todayDt.strftime("%Y%m")
ib = IB()

cDir = "./incremental/"


def fetch_incremental_data(numDays):
    #get last 4 months data for each month
    createFolder(cDir)
    curContrMonth = (todayDt + relativedelta(months=+1)).strftime("%Y%m")
    nextContrMonth = (todayDt + relativedelta(months=+2)).strftime("%Y%m")
    dataDays = []
    for i in range(0,numDays-1):
        dataDays.append((todayDt - relativedelta(days=+i)).strftime("%Y%m%d"))
     


    for day in dataDays:
        dataDay = datetime.strptime(day, "%Y%m%d")

        if dataDay.weekday() >= 5: # 5 and 6 correspond to Saturday and Sunday respectively
            continue


        if (dataDay < todayDt - relativedelta(days=+1)):
            downloadData(cDir, curContrMonth, numDays, day)
            downloadData(cDir, nextContrMonth, numDays, day)


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
        df.set_index('expiry', inplace=True)

        df.to_parquet(path + "cntrMonth-" + month + "-" + day + '.parquet', compression='snappy', index=True)
    except Exception as e:
        print("error in fetching data for " + day)
        print(e)


    



        

def main(numDays=1):

        fetch_incremental_data(numDays)







if __name__ == "__main__":
    import sys
    #main(*sys.argv[1:])
    parser = argparse.ArgumentParser(description="Incremental Downloader")
    parser.add_argument("numDays", type=int, default=1, help="Last how many days to download.")

   

    ib.connect('127.0.0.1', 7497, clientId=0)

    args = parser.parse_args()
    main(args.numDays)

    ib.disconnect()









