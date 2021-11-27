from datetime import date, timedelta
from functools import partial
from time import sleep
from calendar import monthrange
import argparse
import pandas as pd

from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq


def get_last_date_of_month(year: int, month: int) -> date:
    
    return date(year, month, monthrange(year, month)[1])


def convert_dates_to_timeframe(start: date, stop: date) -> str:
    
    return f"{start.strftime('%Y-%m-%d')} {stop.strftime('%Y-%m-%d')}"


def _fetch_data(pytrends, build_payload, timeframe: str) -> pd.DataFrame:
    
    attempts, fetched = 0, False
    while not fetched:
        try:
            build_payload(timeframe=timeframe)
        except ResponseError as err:
            print(err)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break
        else:
            fetched = True
    return pytrends.interest_over_time()


def get_daily_unscaled_data(word: str,
                 start_year: int,
                 start_mon: int,
                 stop_year: int,
                 stop_mon: int,
                 geo: str = 'US',
                 verbose: bool = True,
                 cat:int=0,
                 wait_time: float = 5.0) -> pd.DataFrame:
    
    start_date = date(start_year, start_mon, 1) 
    stop_date = get_last_date_of_month(stop_year, stop_mon)

    pytrends = TrendReq(hl='en-US', tz=360)
    build_payload = partial(pytrends.build_payload,
                            kw_list=[word], cat=cat, geo=geo, gprop='')


    results = {}
    current = start_date
    while current < stop_date:
        last_date_of_month = get_last_date_of_month(current.year, current.month)
        timeframe = convert_dates_to_timeframe(current, last_date_of_month)
        if verbose:
            print(f'{word}:{timeframe}')
        results[current] = _fetch_data(pytrends, build_payload, timeframe)
        current = last_date_of_month + timedelta(days=1)
        sleep(wait_time)  # don't go too fast or Google will send 429s

    daily = pd.concat(results.values()).drop(columns=['isPartial'])

    return daily

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Pytrends data')
    parser.add_argument('-n','--name',type=str,help='Name of the asset that you want to download',default='Apple')
    parser.add_argument('-t','--ticker',type=str,help='Ticker name',default='AAPL')
    parser.add_argument('-say','--start_year',type=int,default=2012)
    parser.add_argument('-sam','--start_month',type=int,default=1)
    parser.add_argument('-soy','--stop_year',type=int,default=2021)
    parser.add_argument('-som','--stop_month',type=int,default=10)
    parser.add_argument('-c','--cat',type=int,help='Category for the asset',default=0)
#     parser.add_argument('-t','--ticker',type=str,help='ticker symbol',default='AAPL')
    args = parser.parse_args()

    name = args.name
    cat = args.cat
    
    df = get_daily_unscaled_data(
        word=name,
        start_year=args.start_year,
        start_mon=args.start_month,
        stop_year=args.stop_year,
        stop_mon=args.stop_month,
        geo='',
        cat=cat,
    )
    df.to_csv(f'Pytrends/{args.ticker}_{cat}.csv')
