import os
import sys
import pickle
import traceback
from typing import Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error as mean_ae, make_scorer

mae = make_scorer(mean_ae)

main_etf = {
    'ApplicationSoftware': 'QQQ',
    'Banks': 'XLF',
    'China': 'KWEB',
    'Oil': 'XOP',
    'RenewableEnergy': 'TAN',
    'Semiconductors': 'SOXL',
    'Gold': 'GDX',
    'DowJones': 'DIA'
}


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
    x_mask = ~((x < x_down) | (x > x_up))

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


def load_sector_data(sector: str) -> pd.DataFrame:
    # if not debug
    if not sys.gettrace():
        path = f'analytics/modeling/'
    # if debug
    else:
        path = f'{os.getcwd()}/../../'

    sector_data = pd.read_csv(filepath_or_buffer=f'{path}'
                                                 f'sectors/'
                                                 f'{sector}/'
                                                 f'datasets/'
                                                 f'data_{sector}.csv')
    return sector_data


def load_traidable_tickers(sector: str) -> list:
    # if not debug
    if not sys.gettrace():
        path = f'analytics/modeling/'
    # if debug
    else:
        path = f'{os.getcwd()}/../../'

    with open(f'{path}'
              f'sectors/'
              f'{sector}/'
              f'tickers/'
              f'traidable_tickers_{sector}.pkl', 'rb') as i:
        traidable_tickers_list = pickle.load(i)

    return traidable_tickers_list


def load_indicators(sector: str) -> list:
    # if not debug
    if not sys.gettrace():
        path = f'analytics/modeling/'
    # if debug
    else:
        path = f'{os.getcwd()}/../../'

    with open(f'{path}'
              f'sectors/'
              f'{sector}/'
              f'tickers/'
              f'indicators_{sector}.pkl', 'rb') as i:
        indicators_list = pickle.load(i)

    return indicators_list


def load_sectors_data(sectors: list) -> dict:
    sectors_data = {}

    for sector in sectors:
        sector_data = load_sector_data(sector)
        traidable_tickers = load_traidable_tickers(sector)
        indicators = load_indicators(sector)
        intersection = \
            list(set(traidable_tickers).intersection(set(indicators)))

        traidable_tickers = [ticker for ticker in traidable_tickers
                             if ticker not in intersection]
        sectors_data[sector] = {
            'data': sector_data,
            'traidable_tickers': traidable_tickers,
            'indicators': indicators
        }

    return sectors_data


def get_indicators_names(indicators: list,
                         features: list) -> list:
    indicators_names = []
    for feature in features:
        indicators_names = indicators_names + [f'{feature}_{indicator}'
                                               for indicator in indicators]
    return indicators_names


class SectorModeler:
    def __init__(self,
                 sectors: list):
        self.sectors_data = load_sectors_data(sectors)

    def run_sector_regression(self,
                              sector: str,
                              features: list):

        df = None
        traidable_tickers = []
        indicators = []

        try:
            df = self.sectors_data[sector]['data']
            traidable_tickers = self.sectors_data[sector]['traidable_tickers']
            indicators = self.sectors_data[sector]['indicators']
        except Exception as e:
            message = f'run_sector_regression error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to get data for sector {sector}')

        if df.empty:
            print(f'Empty dataset for sector: {sector}')

        # Shuffle df and make train/test split
        test_size = 40
        train_df = df[:-test_size]
        test_df = df.tail(40)
        train_df = train_df.sample(frac=1)

        # For future refit
        df = df.sample(frac=1)

        tickers_indicators = {}
        tickers_main_etf = {}
        tickers_models = {}

        main_etf_sector = main_etf[sector]

        for ticker in traidable_tickers:
            ticker_model_dict = {}
            tickers_indicators[ticker] = indicators
            indicators_names = get_indicators_names(indicators=indicators,
                                                    features=features)
            tickers_main_etf[ticker] = main_etf_sector
            main_indicator_name = f'%Gap_{main_etf_sector}'
            target_name = f'%Gap_{ticker}'

            try:
                df_filtered = df[indicators_names + [target_name]]
                df_filtered = \
                    remove_outliers(data_df=df_filtered,
                                    target_column=target_name)
                ticker_target = df_filtered[target_name]

                train_df_filtered = train_df[indicators_names + [target_name]]
                train_df_filtered = \
                    remove_outliers(data_df=train_df_filtered,
                                    target_column=target_name)
                train_df_main_etf_filtered = train_df[[main_indicator_name] + [target_name]]
                train_df_main_etf_filtered = \
                    remove_outliers(data_df=train_df_main_etf_filtered,
                                    target_column=target_name)
                ticker_features_train = train_df_filtered[indicators_names]
                ticker_target_train = train_df_filtered[target_name]
                ticker_features_test = test_df[indicators_names]
                ticker_target_test = test_df[target_name]

                ticker_main_etf_train = train_df_main_etf_filtered[main_indicator_name].to_frame()
                ticker_target_main_etf_train = train_df_main_etf_filtered[target_name]
                ticker_main_etf_test = test_df[main_indicator_name].to_frame()
                ticker_target_main_etf_test = test_df[target_name]

            except Exception as e:
                print(e)
                print(f'Failed to select features for ticker: {ticker}, '
                      f'sector: {sector}')
                print(traidable_tickers)
                print(indicators)
                continue

            # Train on train_data, all features
            lr = RidgeCV(fit_intercept=False,
                         scoring=mae)
            lr.fit(X=ticker_features_train,
                   y=ticker_target_train)

            lr_main_etf = RidgeCV(fit_intercept=False,
                                  scoring=mae)
            lr_main_etf.fit(X=ticker_main_etf_train,
                            y=ticker_target_main_etf_train)

            preds = lr.predict(X=ticker_features_test)
            test_mae = \
                mean_ae(y_true=ticker_target_test, y_pred=preds)

            preds_main_etf = lr_main_etf.predict(X=ticker_main_etf_test)
            test_mae_main_etf = \
                mean_ae(y_true=ticker_target_main_etf_test, y_pred=preds_main_etf)

            print(f'Test MAE for ticker {ticker} is '
                  f'{test_mae}')

            print(f'Test MAE of beta model with main ETF is '
                  f'{test_mae_main_etf}')

            print(f'Beta of {ticker} with main ETF {main_indicator_name} is '
                  f'{lr_main_etf.coef_}')

            ticker_model_dict['model'] = lr
            ticker_model_dict['model_main_etf'] = lr_main_etf
            ticker_model_dict['mae'] = test_mae
            ticker_model_dict['mae_main_etf'] = test_mae_main_etf
            ticker_target_std = ticker_target.std()
            ticker_model_dict['stock_std'] = ticker_target_std
            ticker_target_mean = ticker_target.mean()
            ticker_model_dict['mean'] = ticker_target_mean
            ticker_model_dict['lower_sigma'] = ticker_target_mean - ticker_target_std
            ticker_model_dict['upper_sigma'] = ticker_target_mean + ticker_target_std
            ticker_model_dict['lower_two_sigma'] = ticker_target_mean - 2 * ticker_target_std
            ticker_model_dict['upper_two_sigma'] = ticker_target_mean + 2 * ticker_target_std
            tickers_models[ticker] = ticker_model_dict

        self.sectors_data[sector]['models'] = tickers_models


sectors = ['ApplicationSoftware']
features = ['%Gap']  # , '%YesterdaysGain', '%2DaysGain']
sector_modeler = SectorModeler(sectors=sectors)
sector_modeler.run_sector_regression(sector=sectors[0],
                                     features=features)
