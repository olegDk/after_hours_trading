import sys

import numpy as np
import pandas as pd
import yfinance as yfinance
from tqdm import tqdm
import os
from functools import reduce
from datetime import datetime, timedelta, date
import pickle

cwd = os.getcwd()

if not sys.gettrace():
    sector_stocks = \
        pd.read_csv(filepath_or_buffer=f'{cwd}/analytics/modeling/training/bloomberg_sectors_filtered.csv')
else:
    sector_stocks = \
        pd.read_csv(filepath_or_buffer=f'{cwd}/bloomberg_sectors_filtered.csv')

ETFs = ['SPY', 'QQQ', 'CLOU', 'DIA', 'GDX', 'IWM', 'JETS',
        'SMH', 'TAN', 'XBI', 'XLE', 'XLF', 'XLK', 'XLP',
        'XLU', 'XLV', 'XOP', 'ARKG', 'ARKK', 'EWZ', 'FXI', 'HYG',
        'IEFA', 'IEF', 'EEM', 'EFA', 'IEMG', 'VXX', 'TLT', 'SOXL',
        'XLI', 'XLB', 'XLC', 'XME', 'ITB', 'KWEB', 'CQQQ', 'MCHI', 'BAC']

if not sys.gettrace():
    with open(f'{cwd}/analytics/modeling/all_indicators.pkl', 'wb') as o:
        pickle.dump(ETFs, o)
    with open(f'{cwd}/analytics/modeling/all_indicators.pkl', 'rb') as i:
        ETFs_test = pickle.load(i)
else:
    with open(f'{cwd}/../all_indicators.pkl', 'wb') as o:
        pickle.dump(ETFs, o)
    with open(f'{cwd}/../all_indicators.pkl', 'rb') as i:
        ETFs_test = pickle.load(i)

print(ETFs_test)

ETFs = sorted(ETFs)

BTC_USD = ['BTC-USD']
BTC_F = ['BTC=F']

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
    stocks_df = pd.read_csv(filepath_or_buffer=f'{cwd}/analytics/modeling/'
                                               f'/training/stocks_filtered.csv')
else:
    stocks_df.to_csv(path_or_buf=f'{cwd}/stocks_filtered.csv',
                     index=None)
    stocks_df = pd.read_csv(filepath_or_buffer=f'{cwd}/stocks_filtered.csv')

STOCKs = stocks_df['stock'].to_list()

print(STOCKs)

all_tickers = STOCKs + ETFs + BTC_USD + BTC_F

len(all_tickers)

# Calculating correlations for each sector
# Getting unique sectors
sectors = list(sector_stocks['Sector'].unique())

sector_to_stocks = {sector: sector_stocks[sector_stocks['Sector'] == sector]['Stocks'].values[0].split(',')
                    for sector in sectors}

print(sector_to_stocks)

banks_stocks = sector_to_stocks['Banks']
app_stocks = sector_to_stocks['Application Software']
semi_stocks = sector_to_stocks['Semiconductors']
oil_stocks = sector_to_stocks['Oil']
china_stocks = sector_to_stocks['China']
renew_stocks = sector_to_stocks['Renewable Energy']

tickers_to_update = banks_stocks + app_stocks + semi_stocks + oil_stocks + renew_stocks + china_stocks + ETFs
print(f'Num symbols to update: '
      f'{len(tickers_to_update)}')


def run_data_update():
    for ticker in tqdm(tickers_to_update):
        ticker = ticker.upper()

        start_date = '2019-01-01'
        end_date = date.today().strftime('%Y-%m-%d')
        saved_data = pd.DataFrame()

        try:
            saved_data = pd.read_csv(f'{cwd}/analytics/modeling/training/'
                                     f'ticker_data/ticker_{ticker}.csv', index_col=0)
            last_date = saved_data.index.max()
            start_date = (datetime.strptime(last_date, '%Y-%m-%d')
                          + timedelta(days=1)).date().strftime('%Y-%m-%d')
            end_date = date.today().strftime('%Y-%m-%d')
            print(f'\nTicker {ticker}')
        except Exception as e:
            print(e)
            print(f'Data for ticker {ticker} is missing on the drive')

        print(f'\nDownloading data from date {start_date} until {end_date}')
        data = yfinance.download(tickers=ticker,
                                 start=start_date,
                                 end=end_date,
                                 group_by='Ticker')
        if not data.empty:
            print(f'Saving data for ticker: {ticker}')
            data['ticker'] = ticker
            if not saved_data.empty:
                saved_data = saved_data.append(data)
                saved_data.index = pd.to_datetime(saved_data.index).date
                saved_data = saved_data[~saved_data.index.duplicated(keep='first')]
                saved_data.to_csv(f'{cwd}/analytics/modeling/training/'
                                  f'ticker_data/ticker_{ticker}.csv')
            else:
                data = data[~data.index.duplicated(keep='first')]
                data.to_csv(f'{cwd}/analytics/modeling/training/'
                            f'ticker_data/ticker_{ticker}.csv')
        else:
            print(f'No data for ticker {ticker} on yahoo finance')


def get_data_for_tickers(tickers: list,
                         file_path: str = f'{cwd}/analytics/modeling/training/ticker_data/',
                         file_prefix: str = 'ticker_',
                         calculate_gaps: bool = True,
                         calculate_liquidity: bool = True,
                         filter_tickers: bool = True) -> dict:

    if not file_path[-1:] == '/':
        # include / at the end
        file_path = file_path + '/'

    if sys.gettrace():
        file_path = 'ticker_data/'

    result = {}
    last_possible_date = datetime.now() - timedelta(days=5)  # filter possibly delisted tickers
    first_possible_date = datetime(year=2020,
                                   month=1,
                                   day=1)
    for ticker in tqdm(tickers):
        try:
            data = pd.read_csv(filepath_or_buffer=f'{file_path}{file_prefix}{ticker}.csv')
            if 'Date' not in data.columns:
                data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        except FileNotFoundError:
            print(f'No data for ticker {ticker}')
            continue

        if data.empty:
            print(f'No data for ticker {ticker}')
            continue

        if filter_tickers:
            close_price = data['Adj Close'].tail(1).values[0]
            close_less_than_5 = close_price <= 5
            if close_less_than_5:
                continue
            avg_vol_100 = data['Volume'].tail(100).mean()
            if (avg_vol_100 < 1e6 and close_price < 200) | (avg_vol_100 < 5e5 and close_price >= 200):
                continue
            try:
                market_cap = yfinance.Ticker(ticker=ticker).info['marketCap']
                if market_cap < 2e9:
                    continue
            except Exception as e:
                print(f'No market cap data for ticker {ticker}')
                continue

            last_df_date = data['Date'].tail(1).values[0]
            last_df_date = datetime.strptime(last_df_date, '%Y-%m-%d')
            first_df_date = data['Date'].head(1).values[0]
            first_df_date = datetime.strptime(first_df_date, '%Y-%m-%d')
            if last_df_date < last_possible_date or first_df_date > first_possible_date:
                continue

        if calculate_gaps:
            stock_info = None
            if ticker == 'BTC-USD-NY':
                stock_info = yfinance.Ticker(ticker='BTC-USD')
            else:
                stock_info = yfinance.Ticker(ticker=ticker)
            stock_dividends = stock_info.dividends
            stock_dividends = pd.DataFrame({'Date': stock_dividends.index, 'Dividends': stock_dividends.values})
            stock_dividends['Date'] = stock_dividends['Date'].dt.strftime('%Y-%m-%d')
            data = data.merge(right=stock_dividends,
                              on='Date',
                              how='left',
                              suffixes=('_x', '_y'))
            # Recalculating Adj Close because yfinance Adj Close is not suitable for our setting
            data['Dividends'] = data['Dividends'].shift(-1)
            data['Dividends'].fillna(0, inplace=True)
            data['Adj Close'] = data['Close'] - data['Dividends']
            data['Prev Close'] = data['Adj Close'].shift(1)
            data['PrevPrev Close'] = data['Adj Close'].shift(2)
            data['%Gain'] = (data['Adj Close'] - data['Prev Close']) / data['Prev Close'] * 100
            data['%YesterdaysGain'] = data['%Gain'].shift(1)
            data['%2DaysGain'] = (data['Adj Close'] - data['PrevPrev Close']) / data['PrevPrev Close'] * 100
            data['%Gap'] = (data['Open'] - data['Prev Close']) / data['Prev Close'] * 100

        if calculate_liquidity:
            data['Liquidity'] = np.log((data['Volume'] * data['Adj Close']) / (data['High'] - data['Low']))

        data.dropna(inplace=True)

        result[ticker] = data

    return result


banks_stocks = sector_to_stocks['Banks']
banks_etfs = ['XLF', 'DIA', 'XLI', 'XLE', 'SPY', 'TLT', 'QQQ']
app_stocks = sector_to_stocks['Application Software']
app_etfs = ['QQQ', 'XLK', 'DIA', 'XLF', 'SPY', 'TLT']
semi_stocks = sector_to_stocks['Semiconductors']
semi_etfs = ['QQQ', 'SPY', 'SOXL', 'TLT']
oil_stocks = sector_to_stocks['Oil']
oil_etfs = ['XOP', 'XLE', 'XLF', 'DIA', 'SPY']
renew_stocks = sector_to_stocks['Renewable Energy']
renew_etfs = ['TAN', 'XOP', 'SPY']
china_stocks = sector_to_stocks['China']
china_etfs = ['QQQ', 'KWEB', 'MCHI']

all_sectors = [
    {'sector': 'Banks',
     'data': {
         'stocks': banks_stocks,
         'etfs': banks_etfs
     }},

    {'sector': 'ApplicationSoftware',
     'data': {
         'stocks': app_stocks,
         'etfs': app_etfs
     }},

    {'sector': 'Semiconductors',
     'data': {
         'stocks': semi_stocks,
         'etfs': semi_etfs
     }},

    {'sector': 'Oil',
     'data': {
         'stocks': oil_stocks,
         'etfs': oil_etfs
     }},

    {'sector': 'RenewableEnergy',
     'data': {
         'stocks': renew_stocks,
         'etfs': renew_etfs
     }},

    {'sector': 'China',
     'data': {
         'stocks': china_stocks,
         'etfs': china_etfs
     }}
]

print(all_sectors)


def create_gaps_dataset(targets: list,
                        factors: list) -> pd.DataFrame:
    targets_dict = get_data_for_tickers(tickers=targets)
    factors_dict = get_data_for_tickers(tickers=factors,
                                        filter_tickers=False)

    targets_dict.update(factors_dict)

    dfs_list = []

    for key in list(targets_dict):
        df = targets_dict[key][['Date', '%Gap',
                                '%Gain', '%YesterdaysGain', '%2DaysGain']]
        df.rename(columns={'%Gap': f'%Gap_{key}',
                           '%Gain': f'%Gain_{key}',
                           '%YesterdaysGain': f'%YesterdaysGain_{key}',
                           '%2DaysGain': f'%2DaysGain_{key}'}, inplace=True)
        dfs_list.append(df)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='inner'), dfs_list)
    return df_merged


def create_gaps_data(sectors_names: list):
    for sector_dict in sectors_names:
        sector_name = sector_dict['sector']
        sector_data = sector_dict['data']
        print(f'Creating data for {sector_name}')

        sector_traidable_stocks = sector_data['stocks']
        sector_etfs = sector_data['etfs']

        df = create_gaps_dataset(targets=sector_traidable_stocks,
                                 factors=sector_etfs)

        df.to_csv(path_or_buf=f'{cwd}/analytics/modeling/sectors/{sector_name}/'
                              f'datasets/data_{sector_name}.csv',
                  index=None)

        with open(f'{cwd}/analytics/modeling/sectors/{sector_name}/'
                  f'tickers/traidable_tickers_{sector_name}.pkl',
                  'wb') as output:
            pickle.dump(sector_traidable_stocks, output)

        with open(f'{cwd}/analytics/modeling/sectors/{sector_name}/'
                  f'tickers/indicators_{sector_name}.pkl',
                  'wb') as output:
            pickle.dump(sector_etfs, output)


run_data_update()
create_gaps_data(sectors_names=all_sectors)
