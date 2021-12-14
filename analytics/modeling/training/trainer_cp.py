import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
from multiprocessing import Pool
import pickle
import traceback
import time
import numpy as np
import yfinance

cwd = os.getcwd()

all_etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT']


def get_all_traidable_tickers() -> list:
    if not sys.gettrace():
        stocks_path = f'{cwd}/analytics/modeling/training/all_stocks.csv'
    else:
        stocks_path = f'../training/all_stocks.csv'

    stocks_list = list(pd.read_csv(stocks_path)['Symbol'])

    return stocks_list


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


def get_market_cap_data() -> dict:
    market_cap_data_path = f'analytics/modeling/market_cap_data.pkl'
    if sys.gettrace():
        market_cap_data_path = f'/home/oleh/takion_trader/analytics/' \
                               f'modeling/market_cap_data.pkl'
    with open(market_cap_data_path, 'rb') as i:
        market_cap_data = pickle.load(i)
    return market_cap_data


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
