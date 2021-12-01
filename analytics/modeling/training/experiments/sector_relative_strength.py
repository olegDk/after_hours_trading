import sys
import os
import pickle
from typing import Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sectors_data_path = f'analytics/modeling/sectors'
if sys.gettrace():
    sectors_data_path = f'/home/oleh/takion_trader/analytics/modeling/sectors'


def load_datasets(sectors: list,
                  sectors_columns: dict) -> dict:
    result_dict = {}

    for sector_name in sectors:
        result_dict[sector_name] = {
            'data': pd.read_csv(f'{sectors_data_path}/{sector_name}/datasets/data_{sector_name}.csv',
                                usecols=sectors_columns[sector_name]['data_columns']),
            # 'data': pd.DataFrame(),
            'daily_data': pd.read_csv(f'{sectors_data_path}/{sector_name}/datasets/daily_data_{sector_name}.csv',
                                      usecols=sectors_columns[sector_name]['daily_data_columns'])
        }

    return result_dict


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
                    target_column: str,
                    indicators: list,
                    prefix: str) -> pd.DataFrame:
    df = data_df.copy()
    target_column = f'{prefix}{target_column}'
    target_mask = calculate_outlier_mask(x=df[target_column])
    mask = pd.Series(True, index=df.index)
    indicators_columns = [f'{prefix}{indicator}' for indicator in indicators]
    for column in indicators_columns:
        column_mask = calculate_outlier_mask(x=df[column],
                                             num_iqrs=1)
        mask = mask & column_mask
    mask = ~((mask == True) & (target_mask == False))
    return df[mask]


def leave_outliers(data_df: pd.DataFrame,
                   target_column: str,
                   indicators: list,
                   prefix: str) -> pd.DataFrame:
    df = data_df.copy()
    target_column = f'{prefix}{target_column}'
    target_mask = calculate_outlier_mask(x=df[target_column])
    mask = pd.Series(True, index=df.index)
    indicators_columns = [f'{prefix}{indicator}' for indicator in indicators]
    for column in indicators_columns:
        column_mask = calculate_outlier_mask(x=df[column],
                                             num_iqrs=1.5)
        mask = mask & column_mask
    mask = (mask == True) & (target_mask == False)
    return df[mask]


def get_less_times(start_hour: int,
                   start_minute: int,
                   all_times: list) -> list:
    all_less_times = []
    for hour, minute in all_times:
        if (hour < start_hour) or (minute < start_minute and hour <= start_hour):
            all_less_times = all_less_times + [(hour, minute)]

    return all_less_times


def run_sector_analysis(sector: str,
                        main_sector_etf: str,
                        secondary_etf: str,
                        market_etf: str,
                        stocks_to_load: list = None):
    with open(f'{sectors_data_path}/{sector}/tickers/indicators_{sector}.pkl', 'rb') as i:
        indicators = pickle.load(i)

    if stocks_to_load:
        traidable_tickers = stocks_to_load
    else:
        with open(f'{sectors_data_path}/{sector}/tickers/traidable_tickers_{sector}.pkl', 'rb') as i:
            traidable_tickers = pickle.load(i)

    all_times = []
    for delta_mins in range(0, 60, 5):
        all_times = all_times + [(8, delta_mins)]

    for delta_mins in range(26):
        all_times = all_times + [(9, delta_mins)]

    gaps_prefixes = [f'%Gap_{hour}_{minute}_' for hour, minute in all_times] + ['%Gap_9_30_']
    gaps_columns = []

    for ticker in traidable_tickers + indicators:
        gaps_columns = gaps_columns + [f'{gap_prefix}{ticker}' for gap_prefix in gaps_prefixes]

    cumulative_prem_vol_prop_prefixes = \
        [f'CumulativePremarketVolumeAvgProp_{hour}_{minute}_' for hour, minute in all_times]
    cumulative_prem_vol_prop_columns = []

    for ticker in traidable_tickers + indicators:
        cumulative_prem_vol_prop_columns = \
            cumulative_prem_vol_prop_columns + \
            [f'{c_vol_prop_prefix}{ticker}' for c_vol_prop_prefix in cumulative_prem_vol_prop_prefixes]

    delta_prefixes = []
    delta_columns = []

    for start_hour, start_minute in all_times + [(9, 30)]:
        all_less_times = get_less_times(start_hour=start_hour,
                                        start_minute=start_minute,
                                        all_times=all_times)

        for less_hour, less_minute in all_less_times:
            delta_prefixes = delta_prefixes + \
                             [f'%Delta_{start_hour}_{start_minute}_{less_hour}_{less_minute}_']

    for ticker in traidable_tickers + indicators:
        delta_columns = delta_columns + [f'{delta_prefix}{ticker}' for delta_prefix in delta_prefixes]

    sector_columns = {sector: {'data_columns': ['Date'] +
                                               gaps_columns +
                                               cumulative_prem_vol_prop_columns +
                                               delta_columns,
                               'daily_data_columns': ['Date'] +
                                                     [f'%Gap_{ticker}' for ticker in traidable_tickers] +
                                                     [f'%Gap_{indicator}' for indicator in indicators] +
                                                     [f'%YesterdaysGain_{ticker}' for ticker in traidable_tickers] +
                                                     [f'%YesterdaysGain_{indicator}' for indicator in indicators] +
                                                     [f'%2DaysGain_{ticker}' for ticker in traidable_tickers] +
                                                     [f'%2DaysGain_{indicator}' for indicator in indicators] +
                                                     [f'%5DaysGain_{ticker}' for ticker in traidable_tickers] +
                                                     [f'%5DaysGain_{indicator}' for indicator in indicators],
                               'traidable_tickers': traidable_tickers,
                               'indicators': indicators
                               }}

    if not sector == 'Crypto':
        sector_columns[sector]['daily_data_columns'] = sector_columns[sector]['daily_data_columns'] + \
                                                       [f'%10DaysGain_{ticker}' for ticker in traidable_tickers] + \
                                                       [f'%10DaysGain_{indicator}' for indicator in indicators] + \
                                                       [f'%20DaysGain_{ticker}' for ticker in traidable_tickers] + \
                                                       [f'%20DaysGain_{indicator}' for indicator in indicators] + \
                                                       [f'YesterdaysClose_{ticker}' for ticker in traidable_tickers] + \
                                                       [f'YesterdaysClose_{indicator}' for indicator in indicators] + \
                                                       [f'SMA_20_{ticker}' for ticker in traidable_tickers] + \
                                                       [f'SMA_20_{indicator}' for indicator in indicators] + \
                                                       [f'SMA_8_{ticker}' for ticker in traidable_tickers] + \
                                                       [f'SMA_8_{indicator}' for indicator in indicators]

    datasets = load_datasets(sectors=[sector],
                             sectors_columns=sector_columns)[sector]
    data = datasets['data']
    daily_data = datasets['daily_data']

    # Run relative strength analysis
    # run_relative_strength_analysis(data_df=data,
    #                                daily_data_df=daily_data,
    #                                main_sector_etf=main_sector_etf,
    #                                secondary_etf=secondary_etf,
    #                                market_etf=market_etf)

    # Run daily gains analysis
    # run_daily_gains_analysis(data_df=data, daily_data_df=daily_data, main_sector_etf=main_sector_etf,
    #                          secondary_etf=secondary_etf, market_etf=market_etf, traidable_tickers=traidable_tickers,
    #                          indicators=indicators)

    # Run night buy analysis
    # run_night_buy_analysis(data_df=data, daily_data_df=daily_data, main_sector_etf=main_sector_etf,
    #                        secondary_etf=secondary_etf, market_etf=market_etf, traidable_tickers=traidable_tickers,
    #                        indicators=indicators)

    # Run premarket deltas analysis
    run_premarket_deltas_analysis(data_df=data, daily_data_df=daily_data, main_sector_etf=main_sector_etf,
                                  secondary_etf=secondary_etf, market_etf=market_etf,
                                  traidable_tickers=traidable_tickers,
                                  indicators=indicators)


def run_relative_strength_analysis(data_df: pd.DataFrame,
                                   daily_data_df: pd.DataFrame,
                                   main_sector_etf: str,
                                   secondary_etf: str,
                                   market_etf: str,
                                   traidable_tickers: list):
    data = data_df.copy()
    daily_data = daily_data_df.copy()

    data[f'%Diff_Gap_9_30_{main_sector_etf}_{secondary_etf}'] = \
        data[f'%Gap_9_30_{main_sector_etf}'] - data[f'%Gap_9_30_{secondary_etf}']
    data[f'%Diff_Gap_9_30_{main_sector_etf}_{market_etf}'] = \
        data[f'%Gap_9_30_{main_sector_etf}'] - data[f'%Gap_9_30_{market_etf}']
    data[f'%Diff_Gap_9_30_{secondary_etf}_{market_etf}'] = \
        data[f'%Gap_9_30_{secondary_etf}'] - data[f'%Gap_9_30_{market_etf}']
    data['Avg_Sector_%Gap'] = data[[f'%Gap_9_30_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    data[f'%Diff_Gap_9_30_Avg_Sector_%Gap{main_sector_etf}'] = \
        data['Avg_Sector_%Gap'] - data[f'%Gap_9_30_{main_sector_etf}']
    data[f'%Diff_Gap_9_30_Avg_Sector_%Gap{secondary_etf}'] = \
        data['Avg_Sector_%Gap'] - data[f'%Gap_9_30_{secondary_etf}']
    data[f'%Diff_Gap_9_15_{main_sector_etf}_{secondary_etf}'] = \
        data[f'%Gap_9_15_{main_sector_etf}'] - data[f'%Gap_9_15_{secondary_etf}']
    data['Avg_Sector_%Delta_9_30_9_15'] = \
        data[[f'%Delta_9_30_9_15_{ticker}' for ticker in traidable_tickers]].mean(axis=1)

    daily_data[f'%Diff_Gap_9_30_{main_sector_etf}_{secondary_etf}'] = \
        daily_data[f'%Gap_{main_sector_etf}'] - daily_data[f'%Gap_{secondary_etf}']
    daily_data[f'%Diff_Gap_9_30_{main_sector_etf}_{market_etf}'] = \
        daily_data[f'%Gap_{main_sector_etf}'] - daily_data[f'%Gap_{market_etf}']
    daily_data[f'%Diff_Gap_9_30_{secondary_etf}_{market_etf}'] = \
        daily_data[f'%Gap_{secondary_etf}'] - daily_data[f'%Gap_{market_etf}']
    daily_data['Avg_Sector_%Gap'] = \
        daily_data[[f'%Gap_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    daily_data[f'%Diff_Gap_9_30_Avg_Sector_%Gap{main_sector_etf}'] = \
        daily_data['Avg_Sector_%Gap'] - daily_data[f'%Gap_{main_sector_etf}']
    daily_data[f'%Diff_Gap_9_30_Avg_Sector_%Gap{secondary_etf}'] = \
        daily_data['Avg_Sector_%Gap'] - daily_data[f'%Gap_{secondary_etf}']

    daily_data.plot.scatter(x=f'%Diff_Gap_9_30_{main_sector_etf}_{secondary_etf}',
                            y=f'%Diff_Gap_9_30_Avg_Sector_%Gap{main_sector_etf}')

    daily_data.plot.scatter(x=f'%Diff_Gap_9_30_{secondary_etf}_{market_etf}',
                            y=f'%Diff_Gap_9_30_Avg_Sector_%Gap{main_sector_etf}')

    daily_data.plot.scatter(x=f'%Gap_{secondary_etf}',
                            y=f'%Diff_Gap_9_30_{main_sector_etf}'
                              f'_{secondary_etf}')


def run_daily_gains_analysis(data_df: pd.DataFrame,
                             daily_data_df: pd.DataFrame,
                             main_sector_etf: str,
                             secondary_etf: str,
                             market_etf: str,
                             traidable_tickers: list,
                             indicators: list):
    data = data_df.copy()
    daily_data = daily_data_df.copy()

    stock = 'TEAM'

    daily_data = remove_outliers(data_df=daily_data,
                                 target_column=stock,
                                 indicators=indicators,
                                 prefix='%Gap_')

    daily_data[f'%Diff_Gap_9_30_{stock}_{main_sector_etf}'] = \
        daily_data[f'%Gap_{stock}'] - daily_data[f'%Gap_{main_sector_etf}']
    daily_data[f'%Diff_YesterdaysGain_{stock}_{main_sector_etf}'] = \
        daily_data[f'%YesterdaysGain_{stock}'] - daily_data[f'%YesterdaysGain_{main_sector_etf}']
    daily_data[f'%Diff_2DaysGain_{stock}_{main_sector_etf}'] = \
        daily_data[f'%2DaysGain_{stock}'] - daily_data[f'%2DaysGain_{main_sector_etf}']
    daily_data[f'%Diff_5DaysGain_{stock}_{main_sector_etf}'] = \
        daily_data[f'%5DaysGain_{stock}'] - daily_data[f'%5DaysGain_{main_sector_etf}']
    daily_data[f'%Diff_10DaysGain_{stock}_{main_sector_etf}'] = \
        daily_data[f'%10DaysGain_{stock}'] - daily_data[f'%10DaysGain_{main_sector_etf}']
    daily_data[f'%Diff_20DaysGain_{stock}_{main_sector_etf}'] = \
        daily_data[f'%20DaysGain_{stock}'] - daily_data[f'%20DaysGain_{main_sector_etf}']
    daily_data['Avg_Sector_%Gap'] = \
        daily_data[[f'%Gap_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    daily_data[f'%Diff_SMA_20_YesterdaysClose_{stock}'] = \
        (daily_data[f'SMA_20_{stock}'] - daily_data[f'YesterdaysClose_{stock}']) / \
        daily_data[f'YesterdaysClose_{stock}'] * 100
    daily_data[f'%Diff_SMA_8_YesterdaysClose_{stock}'] = \
        (daily_data[f'SMA_8_{stock}'] - daily_data[f'YesterdaysClose_{stock}']) / \
        daily_data[f'YesterdaysClose_{stock}'] * 100
    daily_data[f'%Diff_SMA_20_YesterdaysClose_{main_sector_etf}'] = \
        (daily_data[f'SMA_20_{main_sector_etf}'] - daily_data[f'YesterdaysClose_{main_sector_etf}']) / \
        daily_data[f'YesterdaysClose_{main_sector_etf}'] * 100
    daily_data[f'%Diff_SMA_8_YesterdaysClose_{main_sector_etf}'] = \
        (daily_data[f'SMA_8_{main_sector_etf}'] - daily_data[f'YesterdaysClose_{main_sector_etf}']) / \
        daily_data[f'YesterdaysClose_{main_sector_etf}'] * 100
    daily_data[f'%Diff_Diff_SMA_20_YesterdaysClose_{stock}_Diff_SMA_20_YesterdaysClose_{main_sector_etf}'] = \
        daily_data[f'%Diff_SMA_20_YesterdaysClose_{stock}'] - \
        daily_data[f'%Diff_SMA_20_YesterdaysClose_{main_sector_etf}']

    # bull_start = datetime(year=2021,
    #                       month=3,
    #                       day=31)
    # bull_end = datetime(year=2021,
    #                     month=5,
    #                     day=3)
    # daily_data = daily_data[(pd.to_datetime(daily_data['Date']) >= bull_start) &
    #                         (pd.to_datetime(daily_data['Date']) <= bull_end)]

    daily_data[(daily_data[f'%Diff_10DaysGain_{stock}_{main_sector_etf}'] > 3) | (
            daily_data[f'%Diff_10DaysGain_{stock}_{main_sector_etf}'] < -3)].plot.scatter(
        x=f'%Diff_10DaysGain_{stock}_{main_sector_etf}',
        y=f'%Diff_Gap_9_30_{stock}_{main_sector_etf}')

    daily_data[(daily_data[
                    f'%Diff_Diff_SMA_20_YesterdaysClose_{stock}_Diff_SMA_20_YesterdaysClose_{main_sector_etf}'] > 0.01) | (
                       daily_data[
                           f'%Diff_Diff_SMA_20_YesterdaysClose_{stock}_Diff_SMA_20_YesterdaysClose_{main_sector_etf}'] < -0.01)].plot.scatter(
        x=f'%Diff_Diff_SMA_20_YesterdaysClose_{stock}_Diff_SMA_20_YesterdaysClose_{main_sector_etf}',
        y=f'%Diff_Gap_9_30_{stock}_{main_sector_etf}')


def run_night_buy_analysis(data_df: pd.DataFrame,
                           daily_data_df: pd.DataFrame,
                           main_sector_etf: str,
                           secondary_etf: str,
                           market_etf: str,
                           traidable_tickers: list,
                           indicators: list):
    data = data_df.copy()
    daily_data = daily_data_df.copy()

    bull_start = datetime(year=2021,
                          month=10,
                          day=1)
    bull_end = datetime(year=2021,
                        month=11,
                        day=21)
    daily_data = daily_data[(pd.to_datetime(daily_data['Date']) >= bull_start) &
                            (pd.to_datetime(daily_data['Date']) <= bull_end)]

    stock = 'AMZN'

    daily_data[f'%Diff_Gap_9_30_{stock}_{main_sector_etf}'] = \
        daily_data[f'%Gap_{stock}'] - daily_data[f'%Gap_{main_sector_etf}']
    daily_data[f'%Diff_YesterdaysGain_{stock}_{main_sector_etf}'] = \
        daily_data[f'%YesterdaysGain_{stock}'] - daily_data[f'%YesterdaysGain_{main_sector_etf}']
    daily_data[f'%Diff_2DaysGain_{stock}_{main_sector_etf}'] = \
        daily_data[f'%2DaysGain_{stock}'] - daily_data[f'%2DaysGain_{main_sector_etf}']

    daily_data.plot.scatter(x=f'%Diff_YesterdaysGain_{stock}_{main_sector_etf}',
                            y=f'%Diff_Gap_9_30_{stock}_{main_sector_etf}')


def run_premarket_deltas_analysis(data_df: pd.DataFrame,
                                  daily_data_df: pd.DataFrame,
                                  main_sector_etf: str,
                                  secondary_etf: str,
                                  market_etf: str,
                                  traidable_tickers: list,
                                  indicators: list):
    data = data_df.copy()
    daily_data = daily_data_df.copy()

    stock = 'FB'

    stocks_aggregates = {}

    for ticker in traidable_tickers:
        stocks_aggregates[ticker] = {
            'mean_%Delta_9_30_9_15': data[f'%Delta_9_30_9_15_{ticker}'].mean(),
            'std_%Delta_9_30_9_15': data[f'%Delta_9_30_9_15_{ticker}'].std(),
            'mean_%Delta_9_15_9_14': data[f'%Delta_9_15_9_14_{ticker}'].mean(),
            'std_%Delta_9_15_9_14': data[f'%Delta_9_15_9_14_{ticker}'].std(),
            'mean_%Delta_9_15_9_13': data[f'%Delta_9_15_9_13_{ticker}'].mean(),
            'std_%Delta_9_15_9_13': data[f'%Delta_9_15_9_13_{ticker}'].std(),
            'mean_%Delta_9_15_9_10': data[f'%Delta_9_15_9_10_{ticker}'].mean(),
            'std_%Delta_9_15_9_10': data[f'%Delta_9_15_9_10_{ticker}'].std(),
            'mean_%Delta_9_15_9_0': data[f'%Delta_9_15_9_0_{ticker}'].mean(),
            'std_%Delta_9_15_9_0': data[f'%Delta_9_15_9_0_{ticker}'].std(),
        }
    data['Avg_Sector_%Delta_9_15_8_0'] = \
        data[[f'%Delta_9_15_8_0_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    data['Avg_Sector_%Delta_9_15_8_30'] = \
        data[[f'%Delta_9_15_8_30_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    data['Avg_Sector_%Delta_9_15_8_45'] = \
        data[[f'%Delta_9_15_8_45_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    data['Avg_Sector_%Delta_9_15_9_0'] = \
        data[[f'%Delta_9_15_9_0_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    data['Avg_Sector_%Delta_9_15_9_10'] = \
        data[[f'%Delta_9_15_9_10_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    data['Avg_Sector_%Delta_9_15_9_13'] = \
        data[[f'%Delta_9_15_9_13_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    data['Avg_Sector_%Delta_9_15_9_14'] = \
        data[[f'%Delta_9_15_9_14_{ticker}' for ticker in traidable_tickers]].mean(axis=1)
    data['Avg_Sector_%Delta_9_30_9_15'] = \
        data[[f'%Delta_9_30_9_15_{ticker}' for ticker in traidable_tickers]].mean(axis=1)

    delta = 0.25

    target_45 = data[((data['Avg_Sector_%Delta_9_15_8_30'] < -delta) | (data['Avg_Sector_%Delta_9_15_8_30'] > delta)) &
                     (data['Avg_Sector_%Delta_9_15_8_30'] < 1.5)][
        'Avg_Sector_%Delta_9_30_9_15']
    target_30 = data[(data['Avg_Sector_%Delta_9_15_8_45'] < -delta) | (data['Avg_Sector_%Delta_9_15_8_45'] > delta) &
                     (data['Avg_Sector_%Delta_9_15_8_45'] < 1.5)][
        'Avg_Sector_%Delta_9_30_9_15']
    target_15 = data[(data['Avg_Sector_%Delta_9_15_9_0'] < -delta) | (data['Avg_Sector_%Delta_9_15_9_0'] > delta) &
                     (data['Avg_Sector_%Delta_9_15_9_0'] < 1.5)][
        'Avg_Sector_%Delta_9_30_9_15']

    feat_45 = data[(data['Avg_Sector_%Delta_9_15_8_30'] < -delta) | (data['Avg_Sector_%Delta_9_15_8_30'] > delta) &
                   (data['Avg_Sector_%Delta_9_15_8_30'] < 1.5)][
        'Avg_Sector_%Delta_9_15_8_30']
    feat_30 = data[(data['Avg_Sector_%Delta_9_15_8_45'] < -delta) | (data['Avg_Sector_%Delta_9_15_8_45'] > delta) &
                   (data['Avg_Sector_%Delta_9_15_8_45'] < 1.5)][
        'Avg_Sector_%Delta_9_15_8_45']
    feat_15 = data[(data['Avg_Sector_%Delta_9_15_9_0'] < -delta) | (data['Avg_Sector_%Delta_9_15_9_0'] > delta) &
                   (data['Avg_Sector_%Delta_9_15_9_0'] < 1.5)][
        'Avg_Sector_%Delta_9_15_9_0']

    print(f'Correlation with 8:30 delta {delta}: {target_45.corr(feat_45)}')
    print(f'Correlation with 8:45 delta {delta}: {target_30.corr(feat_30)}')
    print(f'Correlation with 9:00 delta {delta}: {target_15.corr(feat_15)}')

    print(f'Beta with 8:30 delta {delta}: {get_beta(target_45, feat_45)}')
    print(f'Beta with 8:45 delta {delta}: {get_beta(target_30, feat_30)}')
    print(f'Beta with 9:00 delta {delta}: {get_beta(target_15, feat_15)}')

    # data[(data['Avg_Sector_%Delta_9_15_8_0'] < -delta) | (data['Avg_Sector_%Delta_9_15_8_0'] > delta)].plot.scatter(
    #     x='Avg_Sector_%Delta_9_15_8_0',
    #     y='Avg_Sector_%Delta_9_30_9_15')
    data[(data['Avg_Sector_%Delta_9_15_8_30'] < -delta) | (data['Avg_Sector_%Delta_9_15_8_30'] > delta)].plot.scatter(
        x='Avg_Sector_%Delta_9_15_8_30',
        y='Avg_Sector_%Delta_9_30_9_15')
    data[(data['Avg_Sector_%Delta_9_15_8_45'] < -delta) | (data['Avg_Sector_%Delta_9_15_8_45'] > delta)].plot.scatter(
        x='Avg_Sector_%Delta_9_15_8_45',
        y='Avg_Sector_%Delta_9_30_9_15')
    data[(data['Avg_Sector_%Delta_9_15_9_0'] < -delta) | (data['Avg_Sector_%Delta_9_15_9_0'] > delta)].plot.scatter(
        x='Avg_Sector_%Delta_9_15_9_0',
        y='Avg_Sector_%Delta_9_30_9_15')
    # data[(data['Avg_Sector_%Delta_9_15_9_10'] < -delta) | (data['Avg_Sector_%Delta_9_15_9_10'] > delta)].plot.scatter(
    #     x='Avg_Sector_%Delta_9_15_9_10',
    #     y='Avg_Sector_%Delta_9_30_9_15')
    # data[(data['Avg_Sector_%Delta_9_15_9_13'] < -delta) | (data['Avg_Sector_%Delta_9_15_9_13'] > delta)].plot.scatter(
    #     x='Avg_Sector_%Delta_9_15_9_13',
    #     y='Avg_Sector_%Delta_9_30_9_15')
    # data[(data['Avg_Sector_%Delta_9_15_9_14'] < -delta) | (data['Avg_Sector_%Delta_9_15_9_14'] > delta)].plot.scatter(
    #     x='Avg_Sector_%Delta_9_15_9_14',
    #     y='Avg_Sector_%Delta_9_30_9_15')


def get_beta(a: pd.Series,
             b: pd.Series) -> float:
    cov_matrix = np.cov(a, b)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    return beta


run_sector_analysis(sector='Banks',
                    main_sector_etf='XLF',
                    secondary_etf='TLT',
                    market_etf='SPY',
                    stocks_to_load=['C', 'JPM', 'GS'])
