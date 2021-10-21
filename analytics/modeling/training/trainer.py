import sys
import numpy as np
import pandas as pd
import os
import pickle
from typing import Tuple
from tqdm import tqdm
import traceback
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error as mean_ae, make_scorer, r2_score
from scipy.stats import shapiro, normaltest, chisquare, kstest

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


def check_shapiro(a: np.ndarray) -> bool:
    _, p = shapiro(a)
    if p > 0.05:
        return True
    else:
        return False


def check_pearson(a: np.ndarray) -> bool:
    _, p = normaltest(a)
    if p > 0.05:
        return True
    else:
        return False


def check_chisquare(a: np.ndarray) -> bool:
    _, p = chisquare(a)
    if p > 0.05:
        return True
    else:
        return False


def check_kstest(a: np.ndarray) -> bool:
    _, p = kstest(a, 'norm')
    if p > 0.05:
        return True
    else:
        return False


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
            message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to load tickers for sector: {sector}')

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

        df = None
        try:
            df = pd.read_csv(filepath_or_buffer=f'{sector_dir}/'
                                                f'datasets/'
                                                f'data_{sector}.csv')
        except Exception as e:
            message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to load dataset for sector: {sector}')

        if df.empty:
            print(f'Empty dataset for sector: {sector}')
            continue

        try:
            run_regular_sector_regression(sector=sector,
                                          traidable_tickers=traidable_tickers,
                                          indicators=indicators,
                                          data_df=df)

            run_correlation_analysis(sector=sector,
                                     traidable_tickers=traidable_tickers,
                                     indicators=indicators,
                                     data_df=df)
        except Exception as e:
            message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to run sector regression sector: {sector}')


def calculate_abs_diff_when_bt_mae(targets: np.ndarray,
                                   preds: np.ndarray,
                                   test_mae: float) -> float:

    abs_diffs = list(np.abs(targets-preds))
    abs_diffs_bt_mae = [abs_diff for abs_diff in abs_diffs if abs_diff >= test_mae]

    return float(np.mean(abs_diffs_bt_mae))


def run_regular_sector_regression(sector: str,
                                  traidable_tickers: list,
                                  indicators: list,
                                  data_df: pd.DataFrame):
    df = data_df.copy()
    # Shuffle df and make train/test split
    test_size = 120
    train_df = df[:-test_size]
    test_df = df.tail(test_size)
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
            message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to select features for ticker: {ticker}')
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

        errors = np.sqrt((ticker_target_test - preds) ** 2)
        errors_main_etf = np.sqrt((ticker_target_main_etf_test - preds_main_etf) ** 2)

        ticker_model_dict['model'] = lr
        ticker_model_dict['model_main_etf'] = lr_main_etf
        ticker_model_dict['mae'] = test_mae
        ticker_model_dict['mae_main_etf'] = test_mae_main_etf
        ticker_model_dict['r2_score'] = r2_score(y_true=ticker_target_test, y_pred=preds)
        ticker_model_dict['r2_score_main_etf'] = r2_score(y_true=ticker_target_main_etf_test,
                                                          y_pred=preds_main_etf)
        ticker_model_dict['shapiro_normal'] = check_shapiro(a=errors)
        ticker_model_dict['pearson_normal'] = check_pearson(a=errors)
        ticker_model_dict['ks_normal'] = check_kstest(a=errors)
        ticker_model_dict['chisquare_normal'] = check_chisquare(a=errors)
        ticker_model_dict['shapiro_normal_main_etf'] = check_shapiro(a=errors_main_etf)
        ticker_model_dict['pearson_normal_main_etf'] = check_pearson(a=errors_main_etf)
        ticker_model_dict['ks_normal_main_etf'] = check_kstest(a=errors_main_etf)
        ticker_model_dict['chisquare_normal_main_etf'] = check_chisquare(a=errors_main_etf)
        ticker_model_dict['n_days_error_bt_mae'] = \
            sum(i > test_mae
                for i in list(np.abs(ticker_target_test-preds)))
        ticker_model_dict['n_days_error_bt_main_etf_mae'] = \
            sum(i > test_mae_main_etf
                for i in list(np.abs(ticker_target_main_etf_test - preds_main_etf)))
        ticker_model_dict['mean_abs_diff_when_bt_mae'] = \
            calculate_abs_diff_when_bt_mae(ticker_target_test, preds, test_mae)
        ticker_model_dict['mean_abs_diff_when_bt_main_etf_mae'] = \
            calculate_abs_diff_when_bt_mae(ticker_target_main_etf_test, preds_main_etf, test_mae_main_etf)
        ticker_target_std = ticker_target.std()
        ticker_model_dict['stock_std'] = ticker_target_std
        ticker_target_mean = ticker_target.mean()
        ticker_model_dict['mean'] = ticker_target_mean
        ticker_model_dict['lower_sigma'] = ticker_target_mean - ticker_target_std
        ticker_model_dict['upper_sigma'] = ticker_target_mean + ticker_target_std
        ticker_model_dict['lower_two_sigma'] = ticker_target_mean - 2 * ticker_target_std
        ticker_model_dict['upper_two_sigma'] = ticker_target_mean + 2 * ticker_target_std
        tickers_models[ticker] = ticker_model_dict

        if test_mae <= 1:
            # Refit and save filtered
            lr.fit(X=ticker_features,
                   y=ticker_target)
            lr_main_etf.fit(X=ticker_main_etf,
                            y=ticker_main_etf_target)
            preds_full = lr.predict(ticker_features)
            preds_full_main_etf = lr_main_etf.predict(ticker_main_etf)
            ticker_model_dict['r2_score_full'] = r2_score(y_true=ticker_target,
                                                          y_pred=preds_full)
            ticker_model_dict['r2_score_main_etf_full'] = r2_score(y_true=ticker_main_etf_target,
                                                                   y_pred=preds_full_main_etf)
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

        # Dumping models and data required for live trading
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

        # Dumping data for manual analysis
        if 'statistics' not in os.listdir(sector_path):
            os.mkdir(f'{sector_path}/statistics')

        statistics_path = f'{sector_path}/statistics'

        statistics_fields = ['mae', 'n_days_error_bt_mae',
                             'mae_main_etf', 'n_days_error_bt_main_etf_mae',
                             'stock_std', 'mean', 'lower_sigma', 'upper_sigma',
                             'lower_two_sigma', 'upper_two_sigma', 'mean_abs_diff_when_bt_mae',
                             'mean_abs_diff_when_bt_main_etf_mae',
                             'shapiro_normal', 'shapiro_normal_main_etf',
                             'pearson_normal', 'pearson_normal_main_etf',
                             'ks_normal', 'ks_normal_main_etf',
                             'chisquare_normal', 'chisquare_normal_main_etf',
                             'r2_score', 'r2_score_main_etf', 'r2_score_full',
                             'r2_score_main_etf_full'
                             ]

        statistics_dict = {key_stock: {key_statistics: tickers_models[key_stock][key_statistics]
                                       for key_statistics in tickers_models[key_stock]
                                       if key_statistics in statistics_fields}
                           for key_stock in tickers_models}

        pd.DataFrame.from_dict(statistics_dict, orient='index').\
            to_csv(f'{statistics_path}/modeling_statistics.csv')

    except Exception as e:
        message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
        print(message)
        print(traceback.format_exc())
        print(f'Failed to save data for sector: {sector}')


def run_correlation_analysis(sector: str,
                             traidable_tickers: list,
                             indicators: list,
                             data_df: pd.DataFrame):
    df = data_df.copy()
    df = df.sample(frac=1)

    traidable_tickers_filtered = [traidable_ticker for traidable_ticker in traidable_tickers
                                  if f'%Gap_{traidable_ticker}' in list(df.columns)]

    indicators_filtered = [indicator for indicator in indicators
                           if f'%Gap_{indicator}' in list(df.columns)]

    stocks_cor_matrix = pd.DataFrame(index=traidable_tickers_filtered,
                                     columns=traidable_tickers_filtered)

    stocks_beta_matrix = pd.DataFrame(index=traidable_tickers_filtered,
                                      columns=traidable_tickers_filtered)

    stocks_etfs_cor_matrix = pd.DataFrame(index=traidable_tickers_filtered,
                                          columns=indicators_filtered)

    stocks_etfs_beta_matrix = pd.DataFrame(index=traidable_tickers_filtered,
                                           columns=indicators_filtered)

    for ticker_dependent in tqdm(traidable_tickers_filtered):
        ticker_dependent_name = f'%Gap_{ticker_dependent}'
        for ticker_independent in traidable_tickers_filtered:
            if ticker_dependent == ticker_independent:
                stocks_cor_matrix.loc[ticker_dependent, ticker_independent] = 1
                stocks_beta_matrix.loc[ticker_dependent, ticker_independent] = 1
                continue

            try:
                ticker_independent_name = f'%Gap_{ticker_independent}'
                columns_to_select = [ticker_dependent_name, ticker_independent_name]
                cor, beta = calculate_corr_beta(data_df=df[columns_to_select],
                                                dependent=ticker_dependent_name,
                                                independent=ticker_independent_name)
                stocks_cor_matrix.loc[ticker_dependent, ticker_independent] = cor
                stocks_beta_matrix.loc[ticker_dependent, ticker_independent] = beta
            except Exception as e:
                message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                print(message)
                print(traceback.format_exc())
                print(f'Failed to calculate cor and beta for {ticker_dependent} with '
                      f'{ticker_independent}')

        for indicator in indicators_filtered:
            try:
                indicator_name = f'%Gap_{indicator}'
                columns_to_select = [ticker_dependent_name, indicator_name]
                cor, beta = calculate_corr_beta(data_df=df[columns_to_select],
                                                dependent=ticker_dependent_name,
                                                independent=indicator_name)
                stocks_etfs_cor_matrix.loc[ticker_dependent, indicator] = cor
                stocks_etfs_beta_matrix.loc[ticker_dependent, indicator] = beta
            except Exception as e:
                message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                print(message)
                print(traceback.format_exc())
                print(f'Failed to calculate cor and beta for {ticker_dependent} with '
                      f'{indicator}')

    try:
        if not sys.gettrace():
            sectors_path = f'{cwd}/analytics/modeling/sectors/'
        else:
            sectors_path = f'../sectors/'

        sector_path = f'{sectors_path}{sector}'

        if 'statistics' not in os.listdir(sector_path):
            os.mkdir(f'{sector_path}/statistics')

        statistics_path = f'{sector_path}/statistics'

        stocks_cor_matrix.to_csv(path_or_buf=f'{statistics_path}/stocks_cor_matrix.csv')
        stocks_beta_matrix.to_csv(path_or_buf=f'{statistics_path}/stocks_beta_matrix.csv')

        stocks_etfs_cor_matrix.to_csv(path_or_buf=f'{statistics_path}/stocks_etfs_cor_matrix.csv')
        stocks_etfs_beta_matrix.to_csv(path_or_buf=f'{statistics_path}/stocks_etfs_beta_matrix.csv')

    except Exception as e:
        message = f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
        print(message)
        print(traceback.format_exc())
        print(f'Failed to save statistics data for sector: {sector}')


train_all_models()
