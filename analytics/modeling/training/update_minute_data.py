from datetime import datetime, timedelta
from pytz import timezone
from polygon import RESTClient
import pandas as pd
import traceback
import time
import sys
import os
import pickle
from multiprocessing import Pool

EST = timezone('US/Eastern')
DATETIME_FORMAT = '%Y-%m-%d %H:%M'
MIN_FULL_DAYS = 5
cwd = os.getcwd()


def ts_to_datetime(ts: int) -> str:
    return (datetime.fromtimestamp(ts / 1000.0, tz=EST)).strftime('%Y-%m-%d %H:%M')


key = '80wSvxFpuxhxY3qX_JkWSwT6srzUdsyg'


def get_range_data(from_: str,
                   to: str,
                   ticker: str = 'AAPL',
                   level: int = 1) -> pd.DataFrame:
    with RESTClient(key) as client:
        resp = client.stocks_equities_aggregates(ticker,
                                                 level,
                                                 'minute',
                                                 from_,
                                                 to,
                                                 unadjusted=False)

        print(f'1 minute aggregates for {resp.ticker} between {from_} and {to}.')

        results_processed = {}

        for result in resp.results:
            dt = ts_to_datetime(result['t'])
            result_processed = {'Open': result['o'],
                                'High': result['h'],
                                'Low': result['l'],
                                'Close': result['c'],
                                'Volume': result['v'],
                                'VWAP': 0,
                                'time': result['t']}  # result['vw']}
            results_processed[dt] = result_processed

    return pd.DataFrame.from_dict(results_processed, orient='index')


def get_range_data_wrapper(from_: str,
                           to: str,
                           ticker: str = 'AAPL',
                           level: int = 1,
                           n_retries: int = 2) -> pd.DataFrame:
    for _ in range(n_retries):
        try:
            result = get_range_data(from_=from_,
                                    to=to,
                                    ticker=ticker,
                                    level=level)
            return result
        except Exception as e:
            message = f'Get data for ticker error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to get data for ticker {ticker} in'
                  f' date range {from_} to {to}')

    return pd.DataFrame()


def download_data_for_ticker(ticker: str = 'AAPL',
                             n: int = 90) -> pd.DataFrame:
    date_ranges = get_all_date_ranges(n=n)
    results = []
    for date_range in date_ranges:
        result = get_range_data_wrapper(from_=date_range[0],
                                        to=date_range[1],
                                        ticker=ticker,
                                        level=1
                                        )
        if not result.empty:
            print(f'Got data for ticker {ticker} '
                  f'for range {date_range[0]} '
                  f'to {date_range[1]}')
            results = results + [result]
        else:
            print(f'No data for ticker {ticker} in date range {date_range[0]} '
                  f'to {date_range[1]}')

    if results:
        return pd.concat(results)

    return pd.DataFrame()


def get_all_date_ranges(n: int = 120) -> list:
    if n <= MIN_FULL_DAYS:
        to = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if n == 1:
            from_ = to
        else:
            from_ = (datetime.now() - timedelta(days=n)).strftime('%Y-%m-%d')

        return [(from_, to)]

    else:
        n_min_inclusions = n // MIN_FULL_DAYS
        rest = n - n_min_inclusions * MIN_FULL_DAYS
        date_ranges = []
        from_ = datetime.now() - timedelta(days=n)
        from_str = from_.strftime('%Y-%m-%d')

        for i in range(n_min_inclusions):
            to = from_ + timedelta(days=MIN_FULL_DAYS - 1)
            to_str = to.strftime('%Y-%m-%d')
            date_ranges = date_ranges + [(from_str, to_str)]
            from_ = to + timedelta(days=1)
            from_str = from_.strftime('%Y-%m-%d')

        if rest > 0:
            to = from_ + timedelta(days=rest - 1)
            to_str = to.strftime('%Y-%m-%d')
            date_ranges = date_ranges + [(from_str, to_str)]

        return date_ranges


def get_tickers() -> list:
    if not sys.gettrace():
        sectors_path = f'{cwd}/analytics/modeling/sectors/'
    else:
        sectors_path = f'../sectors/'

    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]

    all_tickers = []

    for sector_dir in sectors_dirs:
        sector = sector_dir.split('/')[-1]
        if 'tickers' not in os.listdir(sector_dir):
            print(f'tickers dir missing '
                  f'for sector: {sector}')
            continue

        tickers_files = os.listdir(f'{sector_dir}/tickers')

        try:
            for f in tickers_files:
                with open(f'{sector_dir}/tickers/{f}', 'rb') as inp:
                    tickers = pickle.load(inp)
                    all_tickers = all_tickers + tickers

        except Exception as e:
            message = f'Get tickers error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to init factors')
            continue

    if not sys.gettrace():
        stocks_path = f'{cwd}/analytics/modeling/training/all_stocks.csv'
    else:
        stocks_path = f'../training/all_stocks.csv'

    stocks_list = list(pd.read_csv(stocks_path)['Symbol'])

    all_tickers = list(set(all_tickers + stocks_list))

    return all_tickers


def get_data_for_tickers(tickers: list,
                         n: int = 90):
    if not sys.gettrace():
        training_path = f'{cwd}/analytics/modeling/training'
    else:
        training_path = f'../training'

    if 'ticker_minute_data' not in os.listdir(training_path):
        os.mkdir(f'{training_path}/ticker_minute_data')

    ticker_minute_data_path = f'{training_path}/ticker_minute_data'

    params_list = [(ticker,
                    ticker_minute_data_path,
                    n) for ticker in tickers]

    with Pool() as p:
        p.starmap(func=get_data_for_ticker,
                  iterable=params_list)


def get_data_for_ticker(ticker: str,
                        ticker_minute_data_path: str,
                        n: int = 90):
    saved_data = pd.DataFrame()
    try:
        saved_data = \
            pd.read_csv(f'{ticker_minute_data_path}/ticker_minute_{ticker}.csv',
                        index_col=0)
        last_date = datetime.strptime(saved_data.index.max(),
                                      DATETIME_FORMAT).date()
        today = datetime.now().date()
        n = (today - last_date).days
        if n <= 2:
            print()
    except FileNotFoundError:
        print(f'Premarket data for ticker {ticker} missing on the drive, downloading data '
              f'for {n} days')

    ticker_data = download_data_for_ticker(ticker=ticker, n=n)

    if not ticker_data.empty:
        if not saved_data.empty:
            saved_data = saved_data.append(ticker_data)
            saved_data = saved_data[~saved_data.index.duplicated(keep='first')]
            saved_data.to_csv(f'{ticker_minute_data_path}/ticker_minute_{ticker}.csv')
        else:
            ticker_data = ticker_data[~ticker_data.index.duplicated(keep='first')]
            ticker_data.to_csv(f'{ticker_minute_data_path}/ticker_minute_{ticker}.csv')
    else:
        print(f'Failed to download premarket data for ticker {ticker} '
              f'for last {n} days')


def rename_data_for_tickers():
    if not sys.gettrace():
        training_path = f'{cwd}/analytics/modeling/training'
    else:
        training_path = f'../training'

    ticker_minute_data_path = f'{training_path}/ticker_minute_data'
    for file in os.listdir(ticker_minute_data_path):
        ticker_name = file.split('.')[0].split('_')[-1]
        print(ticker_name)
        df = pd.read_csv(f'{ticker_minute_data_path}/{file}', index_col=0)
        new_file_name = f'{ticker_minute_data_path}/ticker_minute_{ticker_name}.csv'
        df.to_csv(new_file_name)


tickers = get_tickers()
get_data_for_tickers(tickers=tickers)
