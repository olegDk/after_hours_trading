import sys
import pandas as pd
import os
import pickle
from typing import Tuple
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error as mean_ae, make_scorer

cwd = os.getcwd()
mae = make_scorer(mean_ae)

if not sys.gettrace():
    sector_stocks = \
        pd.read_csv(filepath_or_buffer=f'{cwd}/analytics/modeling/training/bloomberg_sectors_filtered.csv')
else:
    sector_stocks = \
        pd.read_csv(filepath_or_buffer=f'{cwd}/bloomberg_sectors_filtered.csv')

sectors = list(sector_stocks['Sector'].unique())

sector_to_main_etf = {sector: sector_stocks[sector_stocks['Sector'] == sector]['MainETF'].values[0]
                      for sector in sectors}


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
    if not sys.gettrace():
        sectors_path = 'analytics/modeling/sectors'
    else:
        sectors_path = f'{os.getcwd()}/../sectors'
    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]

    print(sectors_dirs)

    # for sector_dir in tqdm(['analytics/modeling/sectors/RenewableEnergy']):
    # for sector_dir in tqdm([sectors_dirs[1]]):
    for sector_dir in tqdm(sectors_dirs):
        print(sector_dir)
        sector = sector_dir.split('/')[-1]

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
        intersection = \
            list(set(traidable_tickers).intersection(set(indicators)))

        traidable_tickers = list(set([ticker for ticker in traidable_tickers
                                      if ticker not in intersection]))

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
    tickers_main_etf = {}
    tickers_main_etf_filtered = {}
    tickers_models = {}
    tickers_models_filtered = {}

    main_etf_sector = sector_to_main_etf[sector]

    for ticker in traidable_tickers:
        ticker_model_dict = {}
        tickers_indicators[ticker] = indicators
        indicators_names = [f'%Gap_{indicator}'
                            for indicator in indicators]
        tickers_main_etf[ticker] = main_etf_sector
        main_indicator_name = f'%Gap_{main_etf_sector}'
        target_name = f'%Gap_{ticker}'

        try:
            # For future refit
            df_filtered = df[indicators_names + [target_name]]
            df_filtered = \
                remove_outliers(data_df=df_filtered,
                                target_column=target_name)
            ticker_features = df_filtered[indicators_names]
            ticker_target = df_filtered[target_name]

            df_main_etf_filtered = df[[main_indicator_name] + [target_name]]
            df_main_etf_filtered = \
                remove_outliers(data_df=df_main_etf_filtered,
                                target_column=target_name)
            ticker_main_etf = df_main_etf_filtered[main_indicator_name].to_frame()
            ticker_main_etf_target = df_main_etf_filtered[target_name]

            # For filtering stocks with high test mae
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
        test_mae =\
            mean_ae(y_true=ticker_target_test, y_pred=preds)

        preds_main_etf = lr_main_etf.predict(X=ticker_main_etf_test)
        test_mae_main_etf =\
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
        ticker_model_dict['lower_two_sigma'] = ticker_target_mean - 2*ticker_target_std
        ticker_model_dict['upper_two_sigma'] = ticker_target_mean + 2*ticker_target_std
        tickers_models[ticker] = ticker_model_dict

        if test_mae <= 1.5:
            # Refit and save filtered
            lr.fit(X=ticker_features,
                   y=ticker_target)
            lr_main_etf.fit(X=ticker_main_etf,
                            y=ticker_main_etf_target)
            ticker_model_dict['model'] = lr
            ticker_model_dict['model_main_etf'] = lr_main_etf
            tickers_indicators_filtered[ticker] = indicators
            tickers_main_etf_filtered[ticker] = main_etf_sector
            tickers_models_filtered[ticker] = ticker_model_dict

    # Saving dicts into models directory for given sector
    try:
        if not sys.gettrace():
            sectors_path = f'{cwd}/analytics/modeling/sectors/'
        else:
            sectors_path = f'../sectors/'

        sector_path = f'{sectors_path}{sector}'

        if 'models' not in os.listdir(sector_path):
            os.mkdir(f'{sector_path}/models')

        models_path = f'{sector_path}/models'

        with open(f'{models_path}/tickers_indicators.pkl', 'wb') as o:
            pickle.dump(tickers_indicators, o)
        with open(f'{models_path}/tickers_main_etf.pkl', 'wb') as o:
            pickle.dump(tickers_main_etf, o)
        with open(f'{models_path}/tickers_models.pkl', 'wb') as o:
            pickle.dump(tickers_models, o)
        with open(f'{models_path}/tickers_indicators_filtered.pkl', 'wb') as o:
            pickle.dump(tickers_indicators_filtered, o)
        with open(f'{models_path}/tickers_main_etf_filtered.pkl', 'wb') as o:
            pickle.dump(tickers_main_etf_filtered, o)
        with open(f'{models_path}/tickers_models_filtered.pkl', 'wb') as o:
            pickle.dump(tickers_models_filtered, o)
    except Exception as e:
        print(e)
        print(f'Failed saving data for sector: {sector}')


train_all_models()
