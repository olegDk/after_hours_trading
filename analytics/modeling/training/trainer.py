import numpy as np
import pandas as pd
import os
import pickle
from typing import Tuple
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error as mean_ae, make_scorer

mae = make_scorer(mean_ae)


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


def train_all_models():
    """
    Train models, save ticker: indicator pairs and ticker: model pairs
    :return: tuple with two dicts:
    one for ticker: indicator pairs, other for ticker: model pairs
    """
    sectors_path = 'analytics/modeling/sectors'
    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]

    print(sectors_dirs)

    # for sector_dir in tqdm(sectors_dirs):
    for sector_dir in tqdm(['analytics/modeling/sectors/RenewableEnergy']):
        print(sector_dir)
        sector = sector_dir.split('/')[-1]
        if 'tickers' not in os.listdir(sector_dir):
            print(f'tickers dir missing '
                  f'for sector: {sector}')
            continue
        if 'datasets' not in os.listdir(sector_dir):
            print(f'datasets dir missing '
                  f'for sector: {sector}')
            continue
        if 'models' not in os.listdir(sector_dir):
            print(f'models dir missing '
                  f'for sector: {sector}')
            continue

        traidable_tickers = []
        indicators = []

        tickers_files = os.listdir(f'{sector_dir}/tickers')
        
        try:
            for f in tickers_files:
                with open(f'{sector_dir}/tickers/{f}', 'rb') as inp:
                    if f.startswith('traidable_tickers'):
                        traidable_tickers = pickle.load(inp)
                    elif f.startswith('indicators'):
                        indicators = pickle.load(inp)
        except Exception as e:
            print(e)
            print(f'Failed to load tickers for sector: {sector}')
            continue
    
        if not traidable_tickers:
            print(f'Traidable tickers missing '
                  f'for sector: {sector}')
            continue
        if not indicators:
            print(f'Indicators missing '
                  f'for sector: {sector}')
            continue

        # Check for intersection, for now, don't trade indicators tickers,
        # so exclude intersection from traidable tickers
        intersection =\
            list(set(traidable_tickers).intersection(set(indicators)))

        traidable_tickers = [ticker for ticker in traidable_tickers
                             if ticker not in intersection]

        try:
            run_sector_regression(sector_dir=sector_dir,
                                  sector=sector,
                                  traidable_tickers=traidable_tickers,
                                  indicators=indicators)
        except Exception as e:
            print(e)
            print(f'Failed regression analysis for sector: {sector}')


def run_sector_regression(sector_dir: str,
                          sector: str,
                          traidable_tickers: list,
                          indicators: list):
    df = None
    try:
        df = pd.read_csv(filepath_or_buffer=f'{sector_dir}/'
                                            f'datasets/'
                                            f'data_{sector}.csv')
    except Exception as e:
        print(e)
        print(f'Failed to load dataset for sector: {sector}')

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
    tickers_indicators_filtered = {}
    tickers_models = {}
    tickers_models_filtered = {}

    for ticker in traidable_tickers:
        ticker_model_dict = {}
        tickers_indicators[ticker] = indicators
        indicators_names = [f'%Gap_{indicator}'
                            for indicator in indicators]
        target_name = f'%Gap_{ticker}'

        try:
            # For future refit
            df_filtered = df[indicators_names + [target_name]]
            df_filtered = \
                remove_outliers(data_df=df_filtered,
                                target_column=target_name)
            ticker_features = df_filtered[indicators_names]
            ticker_target = df_filtered[target_name]

            # For filtering stocks with high test mae
            train_df_filtered = train_df[indicators_names + [target_name]]
            train_df_filtered = \
                remove_outliers(data_df=train_df_filtered,
                                target_column=target_name)
            ticker_features_train = train_df_filtered[indicators_names]
            ticker_target_train = train_df_filtered[target_name]
            ticker_features_test = test_df[indicators_names]
            ticker_target_test = test_df[target_name]
        except Exception as e:
            print(e)
            print(f'Failed to select features for ticker: {ticker}, '
                  f'sector: {sector}')
            print(traidable_tickers)
            print(indicators)
            continue

        # Train on train_data
        lr = RidgeCV(fit_intercept=False,
                     scoring=mae)
        lr.fit(X=ticker_features_train,
               y=ticker_target_train)

        preds = lr.predict(X=ticker_features_test)
        test_mae = mean_ae(y_true=ticker_target_test, y_pred=preds)

        print(f'Test MAE for ticker {ticker} is '
              f'{test_mae}')

        ticker_model_dict['model'] = lr
        ticker_model_dict['mae'] = test_mae
        tickers_models[ticker] = ticker_model_dict

        if test_mae <= 1.5:
            # Refit and save filtered
            lr.fit(X=ticker_features, y=ticker_target)
            ticker_model_dict['model'] = lr
            tickers_indicators_filtered[ticker] = indicators
            tickers_models_filtered[ticker] = ticker_model_dict

    # Saving dicts into models directory for given sector
    try:
        with open(f'analytics/modeling/sectors/'
                  f'{sector}/models/tickers_indicators.pkl', 'wb') as o:
            pickle.dump(tickers_indicators, o)
        with open(f'analytics/modeling/sectors/'
                  f'{sector}/models/tickers_models.pkl', 'wb') as o:
            pickle.dump(tickers_models, o)
        with open(f'analytics/modeling/sectors/'
                  f'{sector}/models/tickers_indicators_filtered.pkl',
                  'wb') as o:
            pickle.dump(tickers_indicators_filtered, o)
        with open(f'analytics/modeling/sectors/'
                  f'{sector}/models/tickers_models_filtered.pkl', 'wb') as o:
            pickle.dump(tickers_models_filtered, o)
    except Exception as e:
        print(e)
        print(f'Failed saving data for sector: {sector}')


train_all_models()
