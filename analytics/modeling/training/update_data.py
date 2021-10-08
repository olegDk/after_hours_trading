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

# Calculating correlations for each sector
# Getting unique sectors
sectors = list(sector_stocks['Sector'].unique())

sector_to_stocks = {sector: sector_stocks[sector_stocks['Sector'] == sector]['Stocks'].values[0].split(',')
                    for sector in sectors}

sector_to_etfs = {sector: sector_stocks[sector_stocks['Sector'] == sector]['Indicators'].values[0].split(',')
                  for sector in sectors}

print(sector_to_stocks)
print(sector_to_etfs)

tickers_to_update = []
ETFs = []
BTC_USD = ['BTC-USD']
BTC_F = ['BTC=F']

for sector in sector_to_stocks.keys():
    tickers_to_update = tickers_to_update + sector_to_stocks[sector]

for sector in sector_to_etfs.keys():
    ETFs = ETFs + sector_to_etfs[sector]

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

tickers_to_update = list(set(tickers_to_update + ETFs))

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
                         file_prefix: str = 'ticker_',
                         calculate_gaps: bool = True,
                         calculate_technicals: bool = False,
                         calculate_liquidity: bool = True,
                         filter_tickers: bool = True) -> dict:
    if sys.gettrace():
        file_path = 'ticker_data/'
    else:
        file_path = f'{cwd}/analytics/modeling/training/ticker_data/'

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

        if calculate_technicals:
            data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
            data['SMA_8'] = data['Adj Close'].rolling(window=8).mean()
            rolling_std_20 = data['Adj Close'].rolling(window=20).std()
            rolling_std_8 = data['Adj Close'].rolling(window=8).std()
            data['SMA_20_upper_sigma'] = data['SMA_20'] + rolling_std_20
            data['SMA_20_lower_sigma'] = data['SMA_20'] - rolling_std_20
            data['SMA_8_upper_sigma'] = data['SMA_8'] + rolling_std_8
            data['SMA_8_lower_sigma'] = data['SMA_8'] - rolling_std_8
            data['SMA_20_upper_two_sigma'] = data['SMA_20'] + 2 * rolling_std_20
            data['SMA_20_lower_two_sigma'] = data['SMA_20'] - 2 * rolling_std_20
            data['SMA_8_upper_two_sigma'] = data['SMA_8'] + 2 * rolling_std_8
            data['SMA_8_lower_two_sigma'] = data['SMA_8'] - 2 * rolling_std_8
            gap_mean = data['%Gap'].mean()
            gap_std = data['%Gap'].std()
            data['Above_upper_sigma'] = (data['%Gap'] > gap_mean + gap_std).astype(int)
            data['Below_lower_sigma'] = (data['%Gap'] < gap_mean - gap_std).astype(int)
            data['Above_upper_two_sigma'] = (data['%Gap'] > gap_mean + 2 * gap_std).astype(int)
            data['Below_lower_two_sigma'] = (data['%Gap'] < gap_mean - 2 * gap_std).astype(int)
            data['%SMA_20_upper_sigma_from_close'] = \
                ((data['SMA_20_upper_sigma'] - data['Adj Close']) / data['Adj Close'] * 100).shift(1)
            data['%SMA_20_lower_sigma_from_close'] = \
                ((data['SMA_20_lower_sigma'] - data['Adj Close']) / data['Adj Close'] * 100).shift(1)
            data['%SMA_20_upper_two_sigma_from_close'] = \
                ((data['SMA_20_upper_two_sigma'] - data['Adj Close']) / data['Adj Close'] * 100).shift(1)
            data['%SMA_20_lower_two_sigma_from_close'] = \
                ((data['SMA_20_lower_two_sigma'] - data['Adj Close']) / data['Adj Close'] * 100).shift(1)

            data['%SMA_20_sigma_interval'] = data['%SMA_20_upper_sigma_from_close'] - \
                                             data['%SMA_20_lower_sigma_from_close']
            data['%SMA_20_two_sigma_interval'] = data['%SMA_20_upper_two_sigma_from_close'] - \
                                                 data['%SMA_20_lower_two_sigma_from_close']

            data['NearBollingerBandsFlag'] = \
                data.apply(lambda r:
                           get_bollinger_flag(r['%Gap'],
                                              r['%SMA_20_upper_sigma_from_close'],
                                              r['%SMA_20_lower_sigma_from_close'],
                                              r['%SMA_20_upper_two_sigma_from_close'],
                                              r['%SMA_20_lower_two_sigma_from_close'],
                                              r['%SMA_20_sigma_interval'],
                                              r['%SMA_20_two_sigma_interval']), axis=1)

        if calculate_liquidity:
            data['Liquidity'] = np.log((data['Volume'] * data['Adj Close']) / (data['High'] - data['Low']))

        data.dropna(inplace=True)

        last_df_date = data['Date'].tail(1).values[0]
        last_df_date = datetime.strptime(last_df_date, '%Y-%m-%d')
        first_df_date = data['Date'].head(1).values[0]
        first_df_date = datetime.strptime(first_df_date, '%Y-%m-%d')
        print(f'Stock: {ticker}')
        print(first_df_date)
        print(last_df_date)

        if last_df_date < last_possible_date or first_df_date > first_possible_date:
            print(f'Removing {ticker} because of missing data')
            continue

        result[ticker] = data

    return result


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


all_sectors = []
for sector in sector_to_stocks:
    if sector in ['Reits', 'Utilities', 'Airlines', 'Steel', 'DowJones']:
        sector_dict = {
            'sector': sector,
            'data': {
                'stocks': sector_to_stocks[sector],
                'etfs': sector_to_etfs[sector]
            }
        }
        all_sectors = all_sectors + [sector_dict]

print(all_sectors)


def create_gaps_dataset(targets: list,
                        factors: list) -> pd.DataFrame:
    targets_dict = get_data_for_tickers(tickers=targets)
    factors_dict = get_data_for_tickers(tickers=factors,
                                        filter_tickers=False)

    targets_dict.update(factors_dict)

    dfs_list = []

    features = ['Date', '%Gap', '%Gain', '%YesterdaysGain', '%2DaysGain']

    for key in list(targets_dict):
        df = targets_dict[key][features]
        rename_columns_dict = {
            feature: f'{feature}_{key}' for feature in features[1:]
        }
        df.rename(columns=rename_columns_dict, inplace=True)
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

        dump_gaps_data(sector_name=sector_name,
                       df=df,
                       sector_traidable_stocks=sector_traidable_stocks,
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

    if 'datasets' not in os.listdir():
        os.mkdir(f'{sector_path}/datasets')

    df.to_csv(path_or_buf=f'{sector_path}/datasets/data_{sector_name}.csv',
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


# run_data_update()
create_gaps_data(sectors_names=all_sectors)
