import sys

import numpy as np
import pandas as pd
import yfinance as yfinance
from tqdm import tqdm
import os
from functools import reduce
from datetime import datetime, timedelta, date
import pickle
from typing import Tuple
from multiprocessing import Pool
import traceback
import time

cwd = os.getcwd()


def get_bloomberg_sectors() -> pd.DataFrame:
    if not sys.gettrace():
        sector_stocks = \
            pd.read_csv(filepath_or_buffer=f'{cwd}/analytics/modeling/training/bloomberg_sectors_filtered.csv')
    else:
        sector_stocks = \
            pd.read_csv(filepath_or_buffer=f'{cwd}/bloomberg_sectors_filtered.csv')

    return sector_stocks


def get_market_cap_data() -> dict:
    market_cap_data_path = f'analytics/modeling/market_cap_data.pkl'
    if sys.gettrace():
        market_cap_data_path = f'/home/oleh/takion_trader/analytics/' \
                               f'modeling/market_cap_data.pkl'
    with open(market_cap_data_path, 'rb') as i:
        market_cap_data = pickle.load(i)
    return market_cap_data


def save_filtered_stocks(sector_stocks: pd.DataFrame):
    # Transforming sector_stocks dataframe to divide into rows by stocks and respective sector
    stocks_df = pd.DataFrame(columns=['stock', 'sector'])

    index = 0
    for i, row in tqdm(sector_stocks.iterrows()):
        stocks_str = row['Stocks']
        sector = row['Sector']
        for stock in stocks_str.split(','):
            stocks_df.loc[index] = [stock, sector]
            index += 1

    if not sys.gettrace():
        stocks_df.to_csv(path_or_buf=f'{cwd}/analytics/modeling/'
                                     f'/training/stocks_filtered.csv',
                         index=None)
    else:
        stocks_df.to_csv(path_or_buf=f'{cwd}/stocks_filtered.csv',
                         index=None)


def init_sector_mappings(sector_stocks: pd.DataFrame) -> Tuple[dict, dict]:
    # Getting unique sectors
    sectors = list(sector_stocks['Sector'].unique())

    sector_to_stocks = {sector: sector_stocks[sector_stocks['Sector'] == sector]['Stocks'].values[0].split(',')
                        for sector in sectors}

    sector_to_etfs = {sector: sector_stocks[sector_stocks['Sector'] == sector]['Indicators'].values[0].split(',')
                      for sector in sectors}

    return sector_to_stocks, sector_to_etfs


def get_tickers_to_update(sector_to_stocks: dict,
                          sector_to_etfs: dict) -> list:
    tickers_to_update = []
    ETFs = []

    for sector in sector_to_stocks.keys():
        tickers_to_update = tickers_to_update + sector_to_stocks[sector]

    for sector in sector_to_etfs.keys():
        ETFs = ETFs + sector_to_etfs[sector]

    if not sys.gettrace():
        with open(f'{cwd}/analytics/modeling/all_indicators.pkl', 'wb') as o:
            pickle.dump(ETFs, o)
    else:
        with open(f'{cwd}/../all_indicators.pkl', 'wb') as o:
            pickle.dump(ETFs, o)

    if not sys.gettrace():
        stocks_path = f'{cwd}/analytics/modeling/training/all_stocks.csv'
    else:
        stocks_path = f'../training/all_stocks.csv'

    stocks_list = list(pd.read_csv(stocks_path)['Symbol'])

    tickers_to_update = list(set(tickers_to_update + ETFs + stocks_list))

    return tickers_to_update


def download_data_from_y_finance(ticker: str,
                                 start: str,
                                 end: str,
                                 n_retries: int = 2,
                                 n_sleeping_seconds: int = 10) -> pd.DataFrame:
    data = pd.DataFrame()
    for i in range(n_retries):
        try:
            data = yfinance.download(tickers=ticker,
                                     start=start,
                                     end=end,
                                     group_by='Ticker',
                                     threads=False)
            return data
        except Exception as e:
            message = f'Get ticker data error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to get data for ticker: {ticker}, sleeping for {n_sleeping_seconds} seconds')
            time.sleep(n_sleeping_seconds)
    return data


def run_data_update(tickers_to_update: list):
    for ticker in tqdm(tickers_to_update):
        ticker = ticker.upper()

        start_date = '2019-01-01'
        end_date = date.today().strftime('%Y-%m-%d')
        saved_data = pd.DataFrame()

        try:
            saved_data = pd.read_csv(f'{cwd}/analytics/modeling/training/'
                                     f'ticker_daily_data/ticker_{ticker}.csv', index_col=0)
            last_date = saved_data.index.max()
            start_date = (datetime.strptime(last_date, '%Y-%m-%d')
                          + timedelta(days=1)).date().strftime('%Y-%m-%d')
            end_date = date.today().strftime('%Y-%m-%d')
            print(f'\nTicker {ticker}')
        except Exception as e:
            print(e)
            print(f'Data for ticker {ticker} is missing on the drive')

        print(f'\nDownloading data from date {start_date} until {end_date}')
        data = download_data_from_y_finance(ticker=ticker,
                                            start=start_date,
                                            end=end_date)
        if not data.empty:
            print(f'Saving data for ticker: {ticker}')
            data['ticker'] = ticker
            if not saved_data.empty:
                saved_data = saved_data.append(data)
                saved_data.index = pd.to_datetime(saved_data.index).date
                saved_data = saved_data[~saved_data.index.duplicated(keep='first')]
                saved_data.to_csv(f'{cwd}/analytics/modeling/training/'
                                  f'ticker_daily_data/ticker_{ticker}.csv')
            else:
                data = data[~data.index.duplicated(keep='first')]
                data.to_csv(f'{cwd}/analytics/modeling/training/'
                            f'ticker_daily_data/ticker_{ticker}.csv')
        else:
            print(f'No data for ticker {ticker} on yahoo finance')


def get_data_for_tickers(tickers: list,
                         market_cap_data: dict,
                         file_prefix: str = 'ticker_',
                         calculate_gaps: bool = True,
                         calculate_bollinger: bool = True,
                         calculate_liquidity: bool = True,
                         filter_by_close_price: bool = True,
                         filter_by_avg_vol: bool = True,
                         filter_by_market_cap: bool = True,
                         filter_by_dates: bool = True) -> dict:
    if sys.gettrace():
        file_path = 'ticker_daily_data/'
    else:
        file_path = f'{cwd}/analytics/modeling/training/ticker_daily_data/'

    result_dict = {}
    last_possible_date = datetime.now() - timedelta(days=7)  # filter possibly delisted tickers
    first_possible_date = last_possible_date - timedelta(days=120)
    params_list = [(ticker,
                    market_cap_data,
                    file_path,
                    first_possible_date,
                    last_possible_date,
                    file_prefix,
                    calculate_gaps,
                    calculate_bollinger,
                    calculate_liquidity,
                    filter_by_close_price,
                    filter_by_avg_vol,
                    filter_by_market_cap,
                    filter_by_dates) for ticker in tickers]

    with Pool() as p:
        result_map = p.starmap(func=get_data_for_ticker,
                               iterable=params_list)
    for ticker_dict in result_map:
        ticker = list(ticker_dict.keys())[0]
        data = list(ticker_dict.values())[0]
        if not data.empty:
            result_dict[ticker] = data

    return result_dict


def get_stock_dividends(ticker: str,
                        n_retries: int = 2,
                        n_sleeping_seconds: int = 10) -> pd.DataFrame:
    stock_dividends = pd.DataFrame()
    for i in range(n_retries):
        try:
            if ticker == 'BTC-USD-NY':
                stock_info = yfinance.Ticker(ticker='BTC-USD')
            else:
                stock_info = yfinance.Ticker(ticker=ticker)
            stock_dividends = stock_info.dividends
            stock_dividends = pd.DataFrame({'Date': stock_dividends.index, 'Dividends': stock_dividends.values})
            stock_dividends['Date'] = stock_dividends['Date'].dt.strftime('%Y-%m-%d')
            return stock_dividends
        except Exception as e:
            message = f'Get dividends for ticker {ticker} error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to get dividends '
                  f'data for ticker: {ticker}, sleeping for {n_sleeping_seconds} seconds')
            time.sleep(n_sleeping_seconds)

    return stock_dividends


def calculate_gaps_and_gains(ticker: str,
                             data_df: pd.DataFrame,
                             calculate_10_days_gain: bool = True,
                             calculate_20_days_gain: bool = True) -> pd.DataFrame:
    data = data_df.copy()
    stock_dividends = get_stock_dividends(ticker=ticker)

    # If there will be error merging - fix to
    # return empty dataframe but with two required columns
    data = data.merge(right=stock_dividends,
                      on='Date',
                      how='left',
                      suffixes=('_x', '_y'))

    # Recalculating Adj Close because yfinance Adj Close is not suitable for our setting
    data['Dividends'] = data['Dividends'].shift(-1)
    data['Dividends'].fillna(0, inplace=True)
    data['Adj Close'] = data['Close'] - data['Dividends']
    data['YesterdaysClose'] = data['Adj Close'].shift(1)
    data['YesterdaysVolume'] = data['Volume'].shift(1)
    data['2DaysAgoClose'] = data['Adj Close'].shift(2)
    data['5DaysAgoClose'] = data['Adj Close'].shift(5)

    data['%YesterdaysGain'] = (data['YesterdaysClose'] - data['2DaysAgoClose']) / data['2DaysAgoClose'] * 100
    data['%2DaysGain'] = \
        (data['YesterdaysClose'] - data['2DaysAgoClose'].shift(1)) / data['2DaysAgoClose'].shift(1) * 100
    data['%5DaysGain'] = \
        (data['YesterdaysClose'] - data['5DaysAgoClose'].shift(1)) / data['5DaysAgoClose'].shift(1) * 100

    data['%Gap'] = (data['Open'] - data['YesterdaysClose']) / data['YesterdaysClose'] * 100

    if calculate_10_days_gain:
        data['10DaysAgoClose'] = data['Adj Close'].shift(10)
        data['%10DaysGain'] = \
            (data['YesterdaysClose'] - data['10DaysAgoClose'].shift(1)) / data['10DaysAgoClose'].shift(1) * 100

    if calculate_20_days_gain:
        data['20DaysAgoClose'] = data['Adj Close'].shift(20)
        data['%20DaysGain'] = \
            (data['YesterdaysClose'] - data['20DaysAgoClose'].shift(1)) / data['20DaysAgoClose'].shift(1) * 100

    return data


def calculate_bollinger_features(data_df: pd.DataFrame) -> pd.DataFrame:
    data = data_df.copy()
    data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
    data['SMA_20'] = data['SMA_20'].shift(1)
    data['SMA_8'] = data['Adj Close'].rolling(window=8).mean()
    data['SMA_8'] = data['SMA_8'].shift(1)
    rolling_std_20 = data['Adj Close'].rolling(window=20).std().shift(1)
    rolling_std_8 = data['Adj Close'].rolling(window=8).std().shift(1)
    data['SMA_20_upper_sigma'] = data['SMA_20'] + rolling_std_20
    data['SMA_20_lower_sigma'] = data['SMA_20'] - rolling_std_20
    data['SMA_8_upper_sigma'] = data['SMA_8'] + rolling_std_8
    data['SMA_8_lower_sigma'] = data['SMA_8'] - rolling_std_8
    data['SMA_20_upper_two_sigma'] = data['SMA_20'] + 2 * rolling_std_20
    data['SMA_20_lower_two_sigma'] = data['SMA_20'] - 2 * rolling_std_20
    data['SMA_8_upper_two_sigma'] = data['SMA_8'] + 2 * rolling_std_8
    data['SMA_8_lower_two_sigma'] = data['SMA_8'] - 2 * rolling_std_8

    # gap_mean = data['%Gap'].mean()
    # gap_std = data['%Gap'].std()
    # data['NearSigmaFlag'] = \
    #     data.apply(lambda r:
    #                get_sigma_flag(r['%Gap'],
    #                               gap_mean,
    #                               gap_std), axis=1)
    #
    data['%SMA_20_upper_sigma_from_yesterdays_close'] = \
        ((data['SMA_20_upper_sigma'] - data['YesterdaysClose']) / data['YesterdaysClose'] * 100)
    data['%SMA_20_lower_sigma_from_yesterdays_close'] = \
        ((data['SMA_20_lower_sigma'] - data['YesterdaysClose']) / data['YesterdaysClose'] * 100)
    data['%SMA_20_upper_two_sigma_from_yesterdays_close'] = \
        ((data['SMA_20_upper_two_sigma'] - data['YesterdaysClose']) / data['YesterdaysClose'] * 100)
    data['%SMA_20_lower_two_sigma_from_yesterdays_close'] = \
        ((data['SMA_20_lower_two_sigma'] - data['YesterdaysClose']) / data['YesterdaysClose'] * 100)

    data['%SMA_20_sigma_interval'] = data['%SMA_20_upper_sigma_from_yesterdays_close'] - \
                                     data['%SMA_20_lower_sigma_from_yesterdays_close']
    data['%SMA_20_two_sigma_interval'] = data['%SMA_20_upper_two_sigma_from_yesterdays_close'] - \
                                         data['%SMA_20_lower_two_sigma_from_yesterdays_close']
    #
    # data['NearBollingerBandsFlag'] = \
    #     data.apply(lambda r:
    #                get_bollinger_flag(r['%Gap'],
    #                                   r['%SMA_20_upper_sigma_from_close'],
    #                                   r['%SMA_20_lower_sigma_from_close'],
    #                                   r['%SMA_20_upper_two_sigma_from_close'],
    #                                   r['%SMA_20_lower_two_sigma_from_close'],
    #                                   r['%SMA_20_sigma_interval'],
    #                                   r['%SMA_20_two_sigma_interval']), axis=1)

    return data


def get_data_for_ticker(ticker: str,
                        market_cap_data: dict,
                        file_path: str,
                        first_possible_date: datetime,
                        last_possible_date: datetime,
                        file_prefix: str = 'ticker_',
                        calculate_gaps: bool = True,
                        calculate_bollinger: bool = True,
                        calculate_liquidity: bool = True,
                        filter_by_close_price: bool = True,
                        filter_by_avg_vol: bool = True,
                        filter_by_market_cap: bool = True,
                        filter_by_dates: bool = True) -> dict:
    try:
        data = pd.read_csv(filepath_or_buffer=f'{file_path}{file_prefix}{ticker}.csv')
        if 'Date' not in data.columns:
            data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    except FileNotFoundError:
        print(f'No data for ticker {ticker}')
        return {ticker: pd.DataFrame()}

    if data.empty:
        print(f'No data for ticker {ticker}')
        return {ticker: pd.DataFrame()}

    close_price = data['Adj Close'].tail(1).values[0]

    if filter_by_close_price:
        if close_price <= 5:
            return {ticker: pd.DataFrame()}
    if filter_by_avg_vol:
        avg_vol_100 = data['Volume'].tail(100).mean()
        if (avg_vol_100 < 1e6 and close_price < 200) | (avg_vol_100 < 5e5 and close_price >= 200):
            return {ticker: pd.DataFrame()}
    if filter_by_market_cap:
        mc_valid = valid_by_market_cap(ticker=ticker,
                                       market_cap_data=market_cap_data)

        if not mc_valid:
            return {ticker: pd.DataFrame()}

    if calculate_gaps:
        if ticker == 'BITO':
            data = calculate_gaps_and_gains(ticker=ticker,
                                            data_df=data,
                                            calculate_10_days_gain=False,
                                            calculate_20_days_gain=False)
        else:
            data = calculate_gaps_and_gains(ticker=ticker,
                                            data_df=data)
        if data.empty:
            return {ticker: pd.DataFrame()}

    if ticker == 'BITO':
        calculate_bollinger = False

    if calculate_bollinger:
        data = calculate_bollinger_features(data_df=data)

    if calculate_liquidity:
        data['Liquidity'] = np.log((data['Volume'] * data['Adj Close']) / (data['High'] - data['Low']))

    data.dropna(inplace=True)

    if not data.empty:
        if filter_by_dates:
            data = filter_by_dates_range(ticker=ticker,
                                         data_df=data,
                                         first_possible_date=first_possible_date,
                                         last_possible_date=last_possible_date)
        if data.empty:
            return {ticker: pd.DataFrame()}
    else:
        return {ticker: pd.DataFrame()}

    return {ticker: data}


def valid_by_market_cap(ticker: str,
                        market_cap_data: dict) -> bool:
    market_cap = market_cap_data.get(ticker)
    if not market_cap:
        try:
            market_cap = yfinance.Ticker(ticker=ticker).info['marketCap']
        except Exception as e:
            message = f'Get market cap data for ticker {ticker} error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            return False
    if market_cap < 2e9 and not ticker == 'CENX':
        return False

    return True


def filter_by_dates_range(ticker: str,
                          data_df: pd.DataFrame,
                          first_possible_date: datetime,
                          last_possible_date: datetime) -> pd.DataFrame:
    data = data_df.copy()

    last_df_date = data['Date'].tail(1).values[0]
    last_df_date = datetime.strptime(last_df_date, '%Y-%m-%d')
    first_df_date = data['Date'].head(1).values[0]
    first_df_date = datetime.strptime(first_df_date, '%Y-%m-%d')
    print(f'Stock: {ticker}')
    print(first_df_date)
    print(last_df_date)

    if last_df_date < last_possible_date or first_df_date > first_possible_date:
        print(f'Removing {ticker} because of missing data')
        return pd.DataFrame()

    return data


def get_sigma_flag(gap,
                   mean,
                   std) -> str:
    if gap >= mean + std:
        return 'Above_upper_sigma'
    elif gap <= mean - std:
        return 'Below_lower_sigma'
    elif gap >= mean + 2 * std:
        return 'Above_upper_two_sigma'
    elif gap <= mean - 2 * std:
        return 'Below_lower_two_sigma'
    else:
        return 'Standard_range'


def get_bollinger_flag(gap,
                       pct_sma_20_upper_sigma,
                       pct_sma_20_lower_sigma,
                       pct_sma_20_upper_two_sigma,
                       pct_sma_20_lower_two_sigma,
                       pct_sma_20_sigma_interval,
                       pct_sma_20_two_sigma_interval) -> str:
    compare_list = [pct_sma_20_upper_sigma,
                    pct_sma_20_lower_sigma,
                    pct_sma_20_upper_two_sigma,
                    pct_sma_20_lower_two_sigma]
    closest = min(compare_list, key=lambda x: abs(x - gap))
    near_interval_fraction = 1 / 5
    near_sigma_interval = pct_sma_20_sigma_interval * near_interval_fraction
    near_two_sigma_interval = pct_sma_20_two_sigma_interval * near_interval_fraction
    if closest == pct_sma_20_upper_sigma:
        delta_abs = abs(gap - pct_sma_20_upper_sigma)
        if delta_abs <= near_sigma_interval:
            return 'NearSMA20UpperSigma'
        elif gap < pct_sma_20_upper_sigma:
            return 'BelowSMA20UpperSigma'
        elif gap >= pct_sma_20_upper_sigma:
            return 'AboveSMA20UpperSigma'

    elif closest == pct_sma_20_lower_sigma:
        delta_abs = abs(gap - pct_sma_20_lower_sigma)
        if delta_abs <= near_sigma_interval:
            return 'NearSMA20LowerSigma'
        elif gap < pct_sma_20_lower_sigma:
            return 'BelowSMA20LowerSigma'
        elif gap >= pct_sma_20_lower_sigma:
            return 'AboveSMA20LowerSigma'

    elif closest == pct_sma_20_upper_two_sigma:
        delta_abs = abs(gap - pct_sma_20_upper_two_sigma)
        if delta_abs <= near_two_sigma_interval:
            return 'NearSMA20UpperTwoSigma'
        elif gap < pct_sma_20_upper_two_sigma:
            return 'BelowSMA20UpperTwoSigma'
        elif gap >= pct_sma_20_upper_two_sigma:
            return 'AboveSMA20UpperTwoSigma'

    elif closest == pct_sma_20_lower_two_sigma:
        delta_abs = abs(gap - pct_sma_20_lower_two_sigma)
        if delta_abs <= near_two_sigma_interval:
            return 'NearSMA20LowerTwoSigma'
        elif gap < pct_sma_20_lower_two_sigma:
            return 'BelowSMA20LowerTwoSigma'
        elif gap >= pct_sma_20_lower_two_sigma:
            return 'AboveSMA20LowerTwoSigma'


def get_all_sectors_data(sector_to_stocks: dict,
                         sector_to_etfs: dict) -> list:
    all_sectors = []
    for sector in sector_to_stocks:
        sector_dict = {
            'sector': sector,
            'data': {
                'stocks': sector_to_stocks[sector],
                'etfs': sector_to_etfs[sector]
            }
        }
        all_sectors = all_sectors + [sector_dict]

    return all_sectors


def create_gaps_dataset(targets: list,
                        factors: list,
                        sector: str) -> Tuple[pd.DataFrame, list]:
    market_cap_data = get_market_cap_data()
    targets_dict = get_data_for_tickers(tickers=targets,
                                        market_cap_data=market_cap_data)
    # Saving only those tickers for which we have training data
    traidable_tickers = list(targets_dict.keys())
    factors_dict = get_data_for_tickers(tickers=factors,
                                        market_cap_data=market_cap_data,
                                        filter_by_close_price=False,
                                        filter_by_avg_vol=False,
                                        filter_by_market_cap=False,
                                        filter_by_dates=False)

    targets_dict.update(factors_dict)

    dfs_list = []

    features = ['Date', '%Gap', 'YesterdaysClose', 'YesterdaysVolume',
                '%YesterdaysGain', '%2DaysGain',
                '%5DaysGain', '%10DaysGain', '%20DaysGain',
                'SMA_20', 'SMA_8',
                '%SMA_20_upper_sigma_from_yesterdays_close',
                '%SMA_20_lower_sigma_from_yesterdays_close',
                '%SMA_20_upper_two_sigma_from_yesterdays_close',
                '%SMA_20_lower_two_sigma_from_yesterdays_close',
                '%SMA_20_sigma_interval',
                '%SMA_20_two_sigma_interval'
                ]

    if sector == 'Crypto':
        features = ['Date', '%Gap', 'YesterdaysClose', 'YesterdaysVolume',
                    '%YesterdaysGain', '%2DaysGain', '%5DaysGain']

    for key in list(targets_dict):
        df = targets_dict[key][features]
        rename_columns_dict = {
            feature: f'{feature}_{key}' for feature in features[1:]
        }
        df.rename(columns=rename_columns_dict, inplace=True)
        dfs_list.append(df)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='inner'), dfs_list)

    return df_merged, traidable_tickers


def create_gaps_data(sectors_names: list):
    for sector_dict in sectors_names:
        sector_name = sector_dict['sector']
        sector_data = sector_dict['data']
        print(f'Creating data for {sector_name}')

        sector_traidable_stocks = sector_data['stocks']
        sector_etfs = sector_data['etfs']

        df, sector_traidable_stocks_filtered = \
            create_gaps_dataset(targets=sector_traidable_stocks,
                                factors=sector_etfs,
                                sector=sector_name)

        dump_gaps_data(sector_name=sector_name,
                       df=df,
                       sector_traidable_stocks=sector_traidable_stocks_filtered,
                       sector_etfs=sector_etfs)


def dump_gaps_data(sector_name: str,
                   df: pd.DataFrame,
                   sector_traidable_stocks: list,
                   sector_etfs: list):
    if not sys.gettrace():
        sectors_path = f'{cwd}/analytics/modeling/sectors/'
    else:
        sectors_path = f'../sectors/'

    if sector_name not in os.listdir(sectors_path):
        os.mkdir(f'{sectors_path}{sector_name}')

    sector_path = f'{sectors_path}{sector_name}'

    if 'datasets' not in os.listdir(sector_path):
        os.mkdir(f'{sector_path}/datasets')

    df.to_csv(path_or_buf=f'{sector_path}/datasets/daily_data_{sector_name}.csv',
              index=None)

    if 'tickers' not in os.listdir(sector_path):
        os.mkdir(f'{sector_path}/tickers')

    with open(f'{sector_path}/'
              f'tickers/traidable_tickers_{sector_name}.pkl',
              'wb') as output:
        pickle.dump(sector_traidable_stocks, output)

    with open(f'{sector_path}/'
              f'tickers/indicators_{sector_name}.pkl',
              'wb') as output:
        pickle.dump(sector_etfs, output)


def get_all_traidable_tickers() -> list:
    if not sys.gettrace():
        sectors_path = f'{cwd}/analytics/modeling/sectors/'
    else:
        sectors_path = f'../sectors/'

    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]

    stocks_list = []

    for sector_dir in sectors_dirs:
        sector = sector_dir.split('/')[-1]

        try:
            with open(f'{sector_dir}/tickers/traidable_tickers_{sector}.pkl', 'rb') as inp:
                stocks = pickle.load(inp)
                stocks_list = stocks_list + stocks

        except Exception as e:
            message = f'Get sector stocks maps error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to get maps for sector: {sector}')
            continue

    return stocks_list


def create_market_cap_data():
    stocks = get_all_traidable_tickers()

    with Pool() as p:
        result_map = p.map(func=get_market_cap_data_for_stock,
                           iterable=stocks)

    if result_map:
        result_dict = {}
        for d in result_map:
            result_dict.update(d)
        if not sys.gettrace():
            modeling_path = f'{cwd}/analytics/modeling/'
        else:
            modeling_path = f'../'

        market_cap_data_path = f'{modeling_path}/market_cap_data.pkl'

        with open(market_cap_data_path, 'wb') as o:
            pickle.dump(result_dict, o)


def get_market_cap_data_for_stock(stock: str,
                                  n_retries: int = 2,
                                  sleeping_seconds: int = 10) -> dict:
    market_cap = 0

    for _ in range(n_retries):
        try:
            market_cap = int(yfinance.Ticker(ticker=stock).info['marketCap'])
            print(f'Market cap for stock: {stock} is {market_cap}')
            return {stock: market_cap}
        except Exception as e:
            message = f'Get market cap data error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to get market cap data for stock: {stock}')
            time.sleep(sleeping_seconds)

    return {stock: market_cap}


sector_stocks = get_bloomberg_sectors()
save_filtered_stocks(sector_stocks=sector_stocks)
sector_to_stocks, sector_to_etfs = init_sector_mappings(sector_stocks=sector_stocks)
tickers_to_update = get_tickers_to_update(sector_to_stocks=sector_to_stocks,
                                          sector_to_etfs=sector_to_etfs)
run_data_update(tickers_to_update=tickers_to_update)
all_sectors_data = get_all_sectors_data(sector_to_stocks=sector_to_stocks,
                                        sector_to_etfs=sector_to_etfs)
create_gaps_data(sectors_names=all_sectors_data)
# create_market_cap_data()
