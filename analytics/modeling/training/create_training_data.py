import sys
import os
import pandas as pd
from functools import reduce
from datetime import datetime, timedelta
from typing import Tuple
import pickle
from multiprocessing import Pool
import traceback
import time
from tqdm import tqdm
import warnings
from polygon import RESTClient
warnings.filterwarnings('ignore')

minute_data_path = f'analytics/modeling/training/ticker_minute_data'
daily_data_path = f'analytics/modeling/training/ticker_daily_data'
sectors_data_path = f'analytics/modeling/sectors'
if sys.gettrace():
    minute_data_path = f'/home/oleh/takion_trader/analytics/modeling/training/ticker_minute_data'
    daily_data_path = f'/home/oleh/takion_trader/analytics/modeling/training/ticker_daily_data'
    sectors_data_path = f'/home/oleh/takion_trader/analytics/modeling/sectors'

key = 'EuMC51EF6YZypR2x5kS__K72C6FKdwhF'


def calculate_premarket_volume(minute_df_data: pd.DataFrame,
                               unique_dates_index_str: list,
                               all_times: list) -> Tuple[dict, dict]:
    result_dict = {}

    for unique_date_str in unique_dates_index_str:
        unique_date = datetime.strptime(unique_date_str, '%Y-%m-%d')
        unique_datetime_open = datetime(year=unique_date.year,
                                        month=unique_date.month,
                                        day=unique_date.day,
                                        hour=9,
                                        minute=30)
        minutes_given_date = minute_df_data[minute_df_data['Date'] == unique_date_str]
        minutes_given_date_premarket = minutes_given_date[minutes_given_date.index < unique_datetime_open]
        if not minutes_given_date_premarket.empty:
            result_dict[unique_date_str] = {
                'total': int(minutes_given_date_premarket['Volume'].sum())
            }
            for time_threshold in all_times:
                unique_datetime_threshold = datetime(year=unique_date.year,
                                                     month=unique_date.month,
                                                     day=unique_date.day,
                                                     hour=time_threshold[0],
                                                     minute=time_threshold[1])
                minutes_given_date_premarket_threshold \
                    = minutes_given_date_premarket[minutes_given_date_premarket.index < unique_datetime_threshold]
                if not minutes_given_date_premarket_threshold.empty:
                    result_dict[unique_date_str][time_threshold] = \
                        minutes_given_date_premarket_threshold['Volume'].sum()
                else:
                    result_dict[unique_date_str][time_threshold] = 0
        else:
            result_dict[unique_date_str] = {
                'total': 0
            }
            result_dict[unique_date_str].update({time_threshold: 0 for time_threshold in all_times})

    totals_dict = \
        {unique_date_str: result_dict[unique_date_str]['total']
         for unique_date_str in result_dict.keys()}

    return result_dict, totals_dict


def get_prev_day_close(ticker: str,
                       date: str,
                       n_retries=2,
                       sleeping_seconds=70) -> float:
    for i in range(n_retries):
        try:
            with RESTClient(key) as client:
                resp = client.stocks_equities_daily_open_close(symbol=ticker,
                                                               date=date)
                return resp.__getattribute__('close')
        except Exception as e:
            message = f'Get close data error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to get close data for stock: {ticker}')
            time.sleep(sleeping_seconds)

    return 0.0


def create_premarket_dataset_for_given_time(hour: int,
                                            minute: int,
                                            minute_df_data: pd.DataFrame,
                                            daily_df_data: pd.DataFrame,
                                            unique_dates_index_data: pd.Series,
                                            unique_dates_index_str: list,
                                            premarket_volume_dict: dict,
                                            average_premarket_volume: float) -> pd.DataFrame:
    minute_df = minute_df_data.copy()
    daily_df = daily_df_data.copy()
    unique_dates_index = unique_dates_index_data.copy()
    unique_starting_datetimes = unique_dates_index.map(lambda d: datetime(year=d.year,
                                                                          month=d.month,
                                                                          day=d.day,
                                                                          hour=hour,
                                                                          minute=minute))
    unique_dates_index_datetimes = list(zip(unique_dates_index_str, unique_starting_datetimes))

    # Getting nearest closest datetime or exactly needed
    starting_datetimes_dataframe = pd.DataFrame()
    for starting_datetime_index, starting_datetime in tqdm(unique_dates_index_datetimes):
        starting_datetime_row = minute_df[minute_df.index == starting_datetime]
        if starting_datetime_row.empty:
            # I.e. finding closest to 9:15 but less than 9:15 for given date
            minute_df_default_index = minute_df.reset_index()
            req_indexes = \
                minute_df_default_index.loc[minute_df_default_index['index'].lt(starting_datetime),
                                            'index']
            if not req_indexes.empty:
                closest_starting_datetime_index = \
                    minute_df_default_index[minute_df_default_index.index == req_indexes.idxmax()]['index']
                closest_starting_datetime64 = \
                    closest_starting_datetime_index.to_frame().values[0][0]
                closest_starting_datetime64_timestamp = \
                    pd.Timestamp(closest_starting_datetime64)
                if closest_starting_datetime64_timestamp.year == starting_datetime.year and \
                        closest_starting_datetime64_timestamp.month == starting_datetime.month and \
                        closest_starting_datetime64_timestamp.day == starting_datetime.day:

                    starting_datetime_row = \
                        minute_df[minute_df.index == closest_starting_datetime64]
                    starting_datetime_row[f'CumulativePremarketVolume'] = \
                        premarket_volume_dict[starting_datetime_index][(hour, minute)]
                    starting_datetime_row[f'CumulativePremarketVolumeAvgProp'] = \
                        starting_datetime_row[f'CumulativePremarketVolume'] / average_premarket_volume
                    starting_datetimes_dataframe = \
                        starting_datetimes_dataframe.append(starting_datetime_row)
                else:
                    print(f'Missing data for given datetime {starting_datetime} on '
                          f'todays premarket, trying to get yesterdays close')

                    prev_day_date = closest_starting_datetime64_timestamp.date().strftime('%Y-%m-%d')

                    try:
                        # Using previous day close
                        prev_day_daily_row = daily_df[daily_df.index == prev_day_date]
                        prev_day_close = prev_day_daily_row['Adj Close'].values[0]
                        starting_datetime_row = pd.DataFrame.from_dict({starting_datetime: {
                            'Open': prev_day_close,
                            'High': prev_day_close,
                            'Low': prev_day_close,
                            'Close': prev_day_close,
                            'Volume': 0,
                            'VWAP': prev_day_close
                        }}, orient='index')
                        starting_datetime_row[f'CumulativePremarketVolume'] = \
                            premarket_volume_dict[starting_datetime_index][(hour, minute)]
                        starting_datetime_row[f'CumulativePremarketVolumeAvgProp'] = \
                            starting_datetime_row[f'CumulativePremarketVolume'] / average_premarket_volume
                        starting_datetimes_dataframe = \
                            starting_datetimes_dataframe.append(starting_datetime_row)
                    except KeyError:
                        ticker = daily_df['Ticker'].values[0]
                        prev_day_close = get_prev_day_close(ticker=ticker,
                                                            date=prev_day_date)
                        if not prev_day_close:
                            print(f'Missing data for given datetime {starting_datetime}')
                        else:
                            starting_datetime_row = pd.DataFrame.from_dict({starting_datetime: {
                                'Open': prev_day_close,
                                'High': prev_day_close,
                                'Low': prev_day_close,
                                'Close': prev_day_close,
                                'Volume': 0,
                                'VWAP': prev_day_close
                            }}, orient='index')
                            starting_datetime_row[f'CumulativePremarketVolume'] = \
                                premarket_volume_dict[starting_datetime_index][(hour, minute)]
                            starting_datetime_row[f'CumulativePremarketVolumeAvgProp'] = \
                                starting_datetime_row[f'CumulativePremarketVolume'] / average_premarket_volume
                            starting_datetimes_dataframe = \
                                starting_datetimes_dataframe.append(starting_datetime_row)
            else:
                print(f'Missing data for given datetime {starting_datetime}')
        else:
            starting_datetime_row[f'CumulativePremarketVolume'] = \
                premarket_volume_dict[starting_datetime_index][(hour, minute)]
            starting_datetime_row[f'CumulativePremarketVolumeAvgProp'] = \
                starting_datetime_row[f'CumulativePremarketVolume'] / average_premarket_volume
            starting_datetimes_dataframe = \
                starting_datetimes_dataframe.append(starting_datetime_row)

    starting_datetimes_dataframe.index = starting_datetimes_dataframe.index.date

    return starting_datetimes_dataframe


def create_premarket_dataset_for_ticker(ticker: str,
                                        all_times: list) -> pd.DataFrame:
    print(f'Creating dataset for ticker: {ticker}')
    minute_df = pd.read_csv(f'{minute_data_path}/ticker_minute_{ticker}.csv', index_col=0)
    daily_df = pd.read_csv(f'{daily_data_path}/ticker_{ticker}.csv', index_col=0)

    minute_df.index = pd.to_datetime(minute_df.index)

    # For every unique date leave only rows which are closest to 09:00 from the less side.
    # One row per date
    all_dates = list(minute_df.index.date)
    unique_dates_index = pd.Series(list(sorted(set(all_dates))))
    unique_dates_index_str = list(pd.to_datetime(unique_dates_index).dt.strftime('%Y-%m-%d').values)
    minute_df['Date'] = minute_df.index.strftime('%Y-%m-%d')
    premarket_volume_dict, premarket_totals_dict = \
        calculate_premarket_volume(minute_df_data=minute_df,
                                   unique_dates_index_str=unique_dates_index_str,
                                   all_times=all_times)

    average_premarket_volume = sum(premarket_totals_dict.values()) / (len(premarket_totals_dict))

    dfs = {}

    for start_hour, start_minute in tqdm(all_times):
        print(f'Hour: {start_hour}, minute: {start_minute}')
        data = create_premarket_dataset_for_given_time(hour=start_hour,
                                                       minute=start_minute,
                                                       minute_df_data=minute_df,
                                                       daily_df_data=daily_df,
                                                       unique_dates_index_data=unique_dates_index,
                                                       unique_dates_index_str=unique_dates_index_str,
                                                       premarket_volume_dict=premarket_volume_dict,
                                                       average_premarket_volume=average_premarket_volume)
        dfs[(start_hour, start_minute)] = data

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'CumulativePremarketVolume',
                'CumulativePremarketVolumeAvgProp']
    dfs_list = []

    for key, df in dfs.items():
        df = df[features]
        rename_columns_dict = {
            feature: f'{feature}_{key[0]}_{key[1]}_{ticker}' for feature in features
        }
        df.rename(columns=rename_columns_dict, inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        dfs_list.append(df)

    df_merged = reduce(lambda left, right: pd.merge(left,
                                                    right,
                                                    on=['Date'],
                                                    how='inner'), dfs_list)

    df_merged['Date'] = pd.to_datetime(df_merged['Date']).dt.strftime('%Y-%m-%d')

    return df_merged


def get_all_times(start_hour: int = 9,
                  start_minute: int = 25) -> list:
    now = datetime.now()

    # Creating now_starting_datetime simply to get hours and minutes
    # of starts for timewindows to calculate premarket moves
    now_starting_datetime = datetime(year=now.year,
                                     month=now.month,
                                     day=now.day,
                                     hour=start_hour,
                                     minute=start_minute)
    all_times = []

    for delta_mins in range(26):
        ago_datetime = now_starting_datetime - timedelta(minutes=delta_mins)
        all_times = all_times + [(ago_datetime.hour, ago_datetime.minute)]

    now_starting_datetime_delta_25 = now_starting_datetime - timedelta(minutes=25)

    for delta_mins in range(5, 65, 5):
        ago_datetime = \
            now_starting_datetime_delta_25 - timedelta(minutes=delta_mins)
        all_times = all_times + [(ago_datetime.hour, ago_datetime.minute)]

    return all_times


def create_premarket_dataset_for_sector(sector: str):
    sector_daily_df = pd.read_csv(f'{sectors_data_path}/{sector}/datasets/daily_data_{sector}.csv',
                                  index_col=0)
    sector_daily_df.reset_index(inplace=True)
    sector_daily_df.rename(columns={'index': 'Date'},
                           inplace=True)

    with open(f'{sectors_data_path}/{sector}/tickers/traidable_tickers_{sector}.pkl', 'rb') as i:
        traidable_tickers = pickle.load(i)

    with open(f'{sectors_data_path}/{sector}/tickers/indicators_{sector}.pkl', 'rb') as i:
        indicators = pickle.load(i)

    start_hour, start_minute = (9, 25)
    all_times = get_all_times(start_hour=start_hour,
                              start_minute=start_minute)

    indicators_dfs = []

    for ticker in tqdm(indicators):
        indicators_dfs = indicators_dfs + \
                         [create_premarket_dataset_for_ticker(ticker=ticker,
                                                              all_times=all_times)]

    indicators_df_merged = reduce(lambda left, right: pd.merge(left,
                                                               right,
                                                               on=['Date'],
                                                               how='inner'),
                                  indicators_dfs)

    traidable_tickers_dfs = []

    for ticker in tqdm(traidable_tickers):
        traidable_tickers_dfs = traidable_tickers_dfs + \
                                [create_premarket_dataset_for_ticker(ticker=ticker,
                                                                     all_times=all_times)]

    traidable_tickers_df_merged = reduce(lambda left, right: pd.merge(left,
                                                                      right,
                                                                      on=['Date'],
                                                                      how='inner'),
                                         traidable_tickers_dfs)

    df_merged = pd.merge(left=traidable_tickers_df_merged,
                         right=indicators_df_merged,
                         on=['Date'],
                         how='inner')

    df_merged_full = pd.merge(left=sector_daily_df,
                              right=df_merged,
                              on=['Date'],
                              how='inner')

    df_final = calculate_relationship_features(df_data=df_merged_full,
                                               traidable_tickers=traidable_tickers,
                                               indicators=indicators,
                                               all_times=all_times)

    dump_sector_data(data_df=df_final,
                     sector=sector)


def dump_sector_data(data_df: pd.DataFrame,
                     sector: str):
    df = data_df.copy()

    sector_path = f'{sectors_data_path}/{sector}'

    datasets_path = f'{sector_path}/datasets'

    df.to_csv(path_or_buf=f'{datasets_path}/data_{sector}.csv',
              index=0)


def calculate_relationship_features(df_data: pd.DataFrame,
                                    traidable_tickers: list,
                                    indicators: list,
                                    all_times: list) -> pd.DataFrame:
    df = df_data.copy()
    all_tickers = traidable_tickers + indicators
    # Calculate premarket deltas
    for ticker in tqdm(all_tickers):
        df.rename(columns={f'%Gap_{ticker}': f'%Gap_9_30_{ticker}'},
                  inplace=True)
        traidable_ticker_flag = ticker in tqdm(traidable_tickers)
        df = calculate_delta_features(data_df=df,
                                      ticker=ticker,
                                      all_times=all_times,
                                      traidable_ticker_flag=traidable_ticker_flag)

    return df


def calculate_delta_features(data_df: pd.DataFrame,
                             ticker: str,
                             all_times: list,
                             traidable_ticker_flag: bool) -> pd.DataFrame:
    df = data_df.copy()
    all_times = all_times + [(9, 30)]
    for start_hour, start_minute in tqdm(all_times):
        if not (start_hour == 9 and start_minute == 30):
            df[f'%Gap_{start_hour}_{start_minute}_{ticker}'] = \
                (df[f'Open_{start_hour}_{start_minute}_{ticker}'] - df[f'YesterdaysClose_{ticker}']) / df[
                    f'YesterdaysClose_{ticker}'] * 100
            df.drop(columns=[f'Open_{start_hour}_{start_minute}_{ticker}',
                             f'High_{start_hour}_{start_minute}_{ticker}',
                             f'Low_{start_hour}_{start_minute}_{ticker}',
                             f'Close_{start_hour}_{start_minute}_{ticker}',
                             f'Volume_{start_hour}_{start_minute}_{ticker}'],
                    inplace=True)

    df.drop(columns=[f'YesterdaysClose_{ticker}',
                     f'YesterdaysVolume_{ticker}'],
            inplace=True)

    for start_hour, start_minute in tqdm(all_times):
        all_less_times = get_less_times(start_hour=start_hour,
                                        start_minute=start_minute,
                                        all_times=all_times)

        for less_hour, less_minute in tqdm(all_less_times):

            df[f'%Delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_{ticker}'] = \
                df[f'%Gap_{start_hour}_{start_minute}_{ticker}'] - df[f'%Gap_{less_hour}_{less_minute}_{ticker}']
            df[f'%Delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_lag_1_{ticker}'] = \
                df[f'%Delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_{ticker}'].shift(1)
            df[f'%Delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_lag_2_{ticker}'] = \
                df[f'%Delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_{ticker}'].shift(2)

            if start_hour == 9 and start_minute == 30:
                df[f'Avg_delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_lag_1_{ticker}'] = \
                    df[
                        f'%Delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_lag_1_{ticker}'].expanding().mean()
            else:
                df[f'Avg_delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_{ticker}'] = \
                    df[f'%Delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_{ticker}'].expanding().mean()

    return df


def get_less_times(start_hour: int,
                   start_minute: int,
                   all_times: list) -> list:
    all_less_times = []
    for hour, minute in all_times:
        if (hour < start_hour) or (minute < start_minute and hour <= start_hour):
            all_less_times = all_less_times + [(hour, minute)]

    return all_less_times


def create_datasets_for_sectors():
    sectors_dirs = \
        [f.path for f in os.scandir(sectors_data_path) if f.is_dir()]

    sectors_names = [sector_dir.split('/')[-1] for sector_dir in sectors_dirs]

    with Pool() as p:
        p.map(func=create_premarket_dataset_for_sector,
              iterable=sectors_names)


create_datasets_for_sectors()

# data.to_csv('/home/oleh/interval_prediction/data.csv', index=0)
#
# from openpyxl.utils.dataframe import dataframe_to_rows
# from openpyxl.workbook import Workbook
#
# wb = Workbook()
# ws = wb.active
#
# rows = dataframe_to_rows(data)
#
# for r_idx, row in enumerate(rows, 1):
#     for c_idx, value in enumerate(row, 1):
#          ws.cell(row=r_idx, column=c_idx, value=value)
#
# wb.save('/home/oleh/interval_prediction/data.xls')
