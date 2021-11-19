import sys
import os
import pickle
import pandas as pd

sectors_data_path = f'analytics/modeling/sectors'
if sys.gettrace():
    sectors_data_path = f'/home/oleh/takion_trader/analytics/modeling/sectors'


def load_datasets(sectors: list) -> dict:
    result_dict = {}
    for sector_name in sectors:
        result_dict[sector_name] = \
            pd.read_csv(f'{sectors_data_path}/{sector_name}/datasets/data_{sector_name}.csv')

    return result_dict


def run_sector_relative_strengh_analysis(sector: str,
                                         main_etf: str,
                                         relative_etf: str):
    data = load_datasets(sectors=[sector])

    with open(f'{sectors_data_path}/{sector}/tickers/traidable_tickers_{sector}.pkl', 'rb') as i:
        traidable_tickers = pickle.load(i)

    with open(f'{sectors_data_path}/{sector}/tickers/indicators_{sector}.pkl', 'rb') as i:
        indicators = pickle.load(i)

    data

    data[f'{main_etf}-{relative_etf}'] =
    print()


run_sector_relative_strengh_analysis(sector='Software',
                                     main_etf='QQQ',
                                     relative_etf='DIA')
