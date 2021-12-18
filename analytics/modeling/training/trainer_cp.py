import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce
from typing import Tuple
from multiprocessing import Pool
import pickle
import traceback
import time
import numpy as np
import json
import yfinance
from pytz import timezone
from typing import Tuple
from tqdm import tqdm

EST = timezone('EST')

cwd = os.getcwd()

all_etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT']


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
    stocks = get_tickers()

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


def get_stocks_cor() -> dict:
    stocks_cor_path = f'analytics/modeling/ClassicPremarket/stocks_cor.pkl'
    if sys.gettrace():
        stocks_cor_path = f'/home/oleh/takion_trader/analytics/' \
                               f'modeling/ClassicPremarket/stocks_cor.pkl'
    with open(stocks_cor_path, 'rb') as i:
        stocks_cor_path = pickle.load(i)
    return stocks_cor_path


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

    if not market_cap:
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
                             data_df: pd.DataFrame) -> pd.DataFrame:
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

    data['%Gap'] = (data['Open'] - data['YesterdaysClose']) / data['YesterdaysClose'] * 100

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
                                            data_df=data)
        else:
            data = calculate_gaps_and_gains(ticker=ticker,
                                            data_df=data)
        if data.empty:
            return {ticker: pd.DataFrame()}

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


def calculate_corr_beta(data_df: pd.DataFrame,
                        dependent: str,
                        independent: str,
                        exclude_outliers: bool = True) -> Tuple[float, float]:
    df = data_df.copy()

    if exclude_outliers:
        # Remove IQR outliers by gap and volume from Y
        # We keep outliers in Y if there are corresponding
        # outliers in X
        df = remove_outliers(data_df=df,
                             target_column=dependent)

    Y_gap = df[dependent]
    X_gap = df[independent]

    # Calculating correlation
    gap_corr_matrix = np.corrcoef(Y_gap, X_gap)
    gap_corr = gap_corr_matrix[0, 1]

    # Calculating beta
    gap_cov_matrix = np.cov(Y_gap, X_gap)
    gap_beta = gap_cov_matrix[0, 1] / gap_cov_matrix[1, 1]

    return gap_corr, gap_beta


def run_correlation_modeling(traidable_tickers: list,
                             indicators: list,
                             data_df: pd.DataFrame):
    df = data_df.copy()
    df = df.sample(frac=1)

    traidable_tickers_filtered = [traidable_ticker for traidable_ticker in traidable_tickers
                                  if f'%Gap_{traidable_ticker}' in list(df.columns)]

    indicators_filtered = [indicator for indicator in indicators
                           if f'%Gap_{indicator}' in list(df.columns)]

    params_list = [(ticker_dependent,
                    traidable_tickers_filtered,
                    indicators_filtered,
                    data_df) for ticker_dependent in traidable_tickers_filtered]

    with Pool() as p:
        starmap_result = p.starmap(func=run_correlation_modeling_for_ticker,
                                   iterable=params_list)

    final_dict = {}

    for result_dict in starmap_result:
        final_dict.update(result_dict)

    dump_correlation_statistics_data(d=final_dict)


def run_correlation_modeling_for_ticker(ticker_dependent: str,
                                        traidable_tickers_filtered: list,
                                        indicators_filtered: list,
                                        data_df: pd.DataFrame) -> dict:
    print(f'Run correlation modeling for ticker: {ticker_dependent}')

    stock_cor = {}
    stock_beta = {}
    stock_etfs_cor = {}
    stock_etfs_beta = {}

    df = data_df.copy()

    ticker_dependent_name = f'%Gap_{ticker_dependent}'
    for ticker_independent in traidable_tickers_filtered:
        if ticker_dependent == ticker_independent:
            stock_cor[ticker_independent] = 1
            stock_beta[ticker_independent] = 1
            continue

        try:
            ticker_independent_name = f'%Gap_{ticker_independent}'
            columns_to_select = [ticker_dependent_name, ticker_independent_name]
            cor, beta = calculate_corr_beta(data_df=df[columns_to_select],
                                            dependent=ticker_dependent_name,
                                            independent=ticker_independent_name)
            stock_cor[ticker_independent] = cor
            stock_beta[ticker_independent] = beta
        except Exception as e:
            message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to calculate cor and beta for {ticker_dependent} with '
                  f'{ticker_independent}')

    for indicator in indicators_filtered:
        if ticker_dependent == indicator:
            stock_etfs_cor[indicator] = 1
            stock_etfs_beta[indicator] = 1
            continue

        try:
            indicator_name = f'%Gap_{indicator}'
            columns_to_select = [ticker_dependent_name, indicator_name]
            cor, beta = calculate_corr_beta(data_df=df[columns_to_select],
                                            dependent=ticker_dependent_name,
                                            independent=indicator_name)
            stock_etfs_cor[indicator] = cor
            stock_etfs_beta[indicator] = beta

        except Exception as e:
            message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to calculate cor and beta for {ticker_dependent} with '
                  f'{indicator}')

    result_dict = {
        ticker_dependent: {
            'stock_cor': stock_cor,
            'stock_beta': stock_beta,
            'stock_etfs_cor': stock_etfs_cor,
            'stock_etfs_beta': stock_etfs_beta
        }
    }

    return result_dict


def dump_correlation_statistics_data(d: dict):
    try:
        if not sys.gettrace():
            modeling_path = f'{cwd}/analytics/modeling'
        else:
            modeling_path = f'..'

        cp_path = f'{modeling_path}/ClassicPremarket'

        if 'ClassicPremarket' not in os.listdir(modeling_path):
            os.mkdir(cp_path)

        with open(f'{cp_path}/stocks_cor.pkl', 'wb') as o:
            pickle.dump(d, o)

    except Exception as e:
        message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
        print(message)
        print(traceback.format_exc())


def get_code_for_minute_charts() -> Tuple[str, str]:
    if sys.gettrace():
        ticker_minute_data_path = f'ticker_minute_data_charts_html'
    else:
        ticker_minute_data_path = f'{cwd}/analytics/modeling/training/ticker_minute_data_charts_html'

    with open(f'{ticker_minute_data_path}/ticker_minute_js_header_code.txt', 'r') as txt_file:
        header = txt_file.read()

    with open(f'{ticker_minute_data_path}/ticker_minute_js_footer_code.txt', 'r') as txt_file:
        footer = txt_file.read()

    return header, footer


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


def dict_to_str(d: dict) -> str:
    data_dict_str = json.dumps(d)
    data_dict_str = data_dict_str.replace('"', '')
    data_dict_c_list = list(data_dict_str)
    data_dict_c_list[-1] = ','
    data_dict_str = ''.join(data_dict_c_list)

    return data_dict_str


def create_js_script_for_minute_chart(ticker: str,
                                      market_cap_data: dict,
                                      header_code: str,
                                      footer_code: str,
                                      stock_cor: dict):
    if sys.gettrace():
        ticker_minute_data_path = f'ticker_minute_data'
        ticker_minute_data_html_path = f'ticker_minute_data_charts_html'
        ticker_daily_data_path = f'ticker_daily_data'
    else:
        ticker_minute_data_path = f'{cwd}/analytics/modeling/training/ticker_minute_data'
        ticker_minute_data_html_path = f'{cwd}/analytics/modeling/training/ticker_minute_data_charts_html'
        ticker_daily_data_path = f'{cwd}/analytics/modeling/training/ticker_daily_data'

    file_path = f'{ticker_minute_data_path}/ticker_minute_{ticker}.csv'
    daily_file_path = f'{ticker_daily_data_path}/ticker_{ticker}.csv'

    try:
        ticker_minute_data = pd.read_csv(file_path)
        ticker_daily_data = pd.read_csv(daily_file_path)
        ticker_minute_data.rename(columns={'Unnamed: 0': 'datetime',
                                           'Open': 'open',
                                           'High': 'high',
                                           'Low': 'low',
                                           'Close': 'close',
                                           'Volume': 'value'}, inplace=True)
        if 'Date' not in ticker_daily_data.columns:
            ticker_daily_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        ticker_minute_data = append_pct_chg_columns(data_df=ticker_minute_data,
                                                    daily_data_df=ticker_daily_data)

        candle_data_dict = ticker_minute_data[['time', 'open', 'high', 'low', 'close']].to_dict(orient='row')
        volume_data_dict = ticker_minute_data[['time', 'value']].to_dict(orient='row')
        pct_chg_data_dict = \
            ticker_minute_data[['time', 'openPctNet', 'highPctNet', 'lowPctNet', 'closePctNet']]. \
                rename(columns={'openPctNet': 'open',
                                'highPctNet': 'high',
                                'lowPctNet': 'low',
                                'closePctNet': 'close'
                                }).to_dict(orient='row')

        candle_data_dict_str = dict_to_str(d=candle_data_dict)
        volume_data_dict_str = dict_to_str(d=volume_data_dict)
        pct_chg_data_dict_str = dict_to_str(d=pct_chg_data_dict)

        etfs_cor = stock_cor.get('stock_etfs_cor', {})
        etfs_beta = stock_cor.get('stock_etfs_beta', {})

        market_cap_data_str = 'var market_cap_data_dict = ' + json.dumps(market_cap_data) + ';'
        stock_etfs_cor_str = 'var stock_cor_dict = ' + json.dumps(etfs_cor) + ';'
        stock_etfs_beta_str = 'var stock_beta_dict = ' + json.dumps(etfs_beta) + ';'

        ticker_minute_data_dict_str = f'{header_code}' + \
                                      f'\nvar ticker = "{ticker}"' + \
                                      f'\ncandleSeries.setData(' + \
                                      f'{candle_data_dict_str}' + \
                                      f']);\n' + \
                                      f'\nvolumeSeries.setData(' + \
                                      f'{volume_data_dict_str}' + \
                                      f']);\n' + \
                                      f'\npctCandleSeries.setData(' + \
                                      f'{pct_chg_data_dict_str}' + \
                                      f']);\n' + \
                                      f'{market_cap_data_str}' + \
                                      f'{stock_etfs_cor_str}' + \
                                      f'{stock_etfs_beta_str}' + \
                                      f'{footer_code}'
        with open(f'{ticker_minute_data_html_path}/minute_chart_{ticker}.html', 'w') as txt_file:
            txt_file.write(ticker_minute_data_dict_str)
    except Exception as e:
        message = f'Create chart for ticker {ticker} error: ' \
                  f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
        print(message)
        print(traceback.format_exc())


def create_charts_for_all_stocks():
    tickers = get_tickers()
    market_cap_data = get_market_cap_data()
    stocks_cor = get_stocks_cor()
    header_code, footer_code = get_code_for_minute_charts()
    if not sys.gettrace():
        training_path = f'{cwd}/analytics/modeling/training'
    else:
        training_path = f'../training'

    if 'ticker_minute_data_charts_html' not in os.listdir(training_path):
        os.mkdir(f'{training_path}/ticker_minute_data_charts_html')

    params_list = [(ticker,
                    market_cap_data,
                    header_code,
                    footer_code,
                    stocks_cor.get(ticker, {})) for ticker in tickers]

    with Pool() as p:
        p.starmap(func=create_js_script_for_minute_chart,
                  iterable=params_list)


def append_pct_chg_columns(data_df: pd.DataFrame,
                           daily_data_df: pd.DataFrame) -> pd.DataFrame:
    data = data_df.copy()
    daily_data = daily_data_df.copy()

    data['openPctNet'] = 0
    data['highPctNet'] = 0
    data['lowPctNet'] = 0
    data['closePctNet'] = 0

    all_dates = pd.to_datetime(data['datetime']).dt.strftime('%Y-%m-%d')
    data['date'] = all_dates
    unique_dates_str = list(sorted(set(all_dates)))
    date_to_prev_date = dict(zip(unique_dates_str[1:], unique_dates_str))

    for cur_date in tqdm(date_to_prev_date.keys()):
        prev_date = date_to_prev_date.get(cur_date)
        if not prev_date:
            continue
        prev_day_row = daily_data[daily_data['Date'] == prev_date]
        if prev_day_row.empty:
            continue
        prev_close = prev_day_row['Adj Close'].values[0]

        openPctNet = (data[data['date'] == cur_date]['open'] - prev_close) / prev_close * 100
        highPctNet = (data[data['date'] == cur_date]['high'] - prev_close) / prev_close * 100
        lowPctNet = (data[data['date'] == cur_date]['low'] - prev_close) / prev_close * 100
        closePctNet = (data[data['date'] == cur_date]['close'] - prev_close) / prev_close * 100

        data.loc[data['date'] == cur_date, 'openPctNet'] = openPctNet
        data.loc[data['date'] == cur_date, 'highPctNet'] = highPctNet
        data.loc[data['date'] == cur_date, 'lowPctNet'] = lowPctNet
        data.loc[data['date'] == cur_date, 'closePctNet'] = closePctNet

    return data


def calculate_IQR(x: pd.Series) -> Tuple[float, float, float]:
    x_q1 = x.quantile(0.25)
    x_q3 = x.quantile(0.75)
    x_iqr = x_q3 - x_q1
    return x_iqr, x_q1, x_q3


def calculate_outlier_mask(x: pd.Series,
                           num_iqrs: float = 1.5) -> pd.Series:
    IQR, Q1, Q3 = calculate_IQR(x)
    x_up = Q3 + num_iqrs * IQR
    x_down = Q1 - num_iqrs * IQR
    x_mask = ~((x < x_down) | (x > x_up))  # valid observations

    return x_mask


def remove_outliers(data_df: pd.DataFrame,
                    target_column: str) -> pd.DataFrame:
    df = data_df.copy()
    target_mask = calculate_outlier_mask(x=df[target_column])
    mask = pd.Series(True, index=df.index)
    for column in list(df.columns):
        if "%Gap" in column and column != target_column:
            column_mask = calculate_outlier_mask(x=df[column],
                                                 num_iqrs=1.5)
            mask = mask & column_mask
    mask = ~((mask == True) & (target_mask == False))
    return df[mask]


def leave_outliers(data_df: pd.DataFrame,
                   target_column: str) -> pd.DataFrame:
    df = data_df.copy()
    target_mask = calculate_outlier_mask(x=df[target_column])
    mask = pd.Series(True, index=df.index)
    for column in list(df.columns):
        if "%Gap" in column and column != target_column:
            column_mask = calculate_outlier_mask(x=df[column],
                                                 num_iqrs=1.5)
            mask = mask & column_mask
    mask = (mask == True) & (target_mask == False)
    return df[mask]


def run_cp_training():
    traidable_tickers = get_tickers()
    # traidable_tickers = ['JWN', 'NET', 'AAPL', 'MA']
    indicators = ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT']

    market_cap_data = get_market_cap_data()

    tickers_dfs = get_data_for_tickers(tickers=traidable_tickers,
                                       filter_by_market_cap=True,
                                       market_cap_data=market_cap_data)

    indicators_dfs = get_data_for_tickers(tickers=indicators,
                                          filter_by_market_cap=False,
                                          market_cap_data=market_cap_data)

    tickers_dfs.update(indicators_dfs)

    features = ['Date', '%Gap']

    dfs_list = []

    for key in list(tickers_dfs.keys()):
        df = tickers_dfs[key][features]
        rename_columns_dict = {
            feature: f'{feature}_{key}' for feature in features[1:]
        }
        df.rename(columns=rename_columns_dict, inplace=True)
        dfs_list.append(df)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='inner'), dfs_list)

    try:
        run_correlation_modeling(traidable_tickers=traidable_tickers,
                                 indicators=indicators,
                                 data_df=df_merged)
    except Exception as e:
        message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
        print(message)
        print(traceback.format_exc())
        print(f'Failed to run classic premarket training')


# create_market_cap_data()
# run_cp_training()
# create_charts_for_all_stocks()
market_cap_data = get_market_cap_data()
stocks_cor = get_stocks_cor()
header_code, footer_code = get_code_for_minute_charts()
create_js_script_for_minute_chart(ticker='IWM',
                                  header_code=header_code,
                                  footer_code=footer_code,
                                  stock_cor=stocks_cor['IWM'],
                                  market_cap_data=market_cap_data)

