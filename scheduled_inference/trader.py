import os
import pickle
from typing import Tuple
from rabbit_sender import RabbitSender

RABBIT_MQ_HOST = 'rabbit'
RABBIT_MQ_PORT = 5672
ORDER_RELATED_DATA = 'orderRelatedData'


class Trader:
    def __init__(self):
        self.__factors, self.__stock_to_sector,\
            self.__sector_to_indicators = self.__init_factors()
        print(self.__factors)
        print(self.__stock_to_sector)
        print(self.__sector_to_indicators)
        self.__models = self.__init_models()
        self.__rabbit = RabbitSender(RABBIT_MQ_HOST, RABBIT_MQ_PORT)

    def __init_factors(self) -> Tuple[dict, dict, dict]:
        print('Initializing factors...')
        sectors_path = 'modeling/sectors'
        sectors_dirs = \
            [f.path for f in os.scandir(sectors_path) if f.is_dir()]

        factors_dict = {}
        stock_to_sector = {}
        sector_to_indicators = {}
        for sector_dir in sectors_dirs:
            sector = sector_dir.split('/')[-1]
            print(f'Sector: {sector}...')
            tickers_indicators_filtered_path\
                = f'{sector_dir}/models/tickers_indicators_filtered.pkl'
            try:
                with open(tickers_indicators_filtered_path, 'rb') as i:
                    tickers_indicators_filtered = pickle.load(i)
                factors_dict.update(tickers_indicators_filtered)
                current_sector_stocks = list(tickers_indicators_filtered.keys())
                key = current_sector_stocks[0]
                current_sector_indicators = tickers_indicators_filtered[key]
                sector_to_indicators[sector] = current_sector_indicators
                for stock in current_sector_stocks:
                    stock_to_sector[stock] = sector
                print(f'Sector: {sector} loaded.')
                print(f'-------')
            except Exception as e:
                print(e)
                print(f'Failed to load tickers for sector: {sector}')
                continue

        return factors_dict, stock_to_sector, sector_to_indicators

    def __init_models(self) -> dict:
        print('Initializing models...')
        sectors_path = 'modeling/sectors'
        sectors_dirs = \
            [f.path for f in os.scandir(sectors_path) if f.is_dir()]

        models_dict = {}
        for sector_dir in sectors_dirs:
            sector = sector_dir.split('/')[-1]
            print(f'Sector: {sector}...')
            tickers_models_filtered_path \
                = f'{sector_dir}/models/tickers_models_filtered.pkl'
            try:
                with open(tickers_models_filtered_path, 'rb') as i:
                    tickers_models_filtered = pickle.load(i)
                models_dict.update(tickers_models_filtered)
                print(f'Sector: {sector} loaded.')
                print(f'-------')
            except Exception as e:
                print(e)
                print(f'Failed to load tickers for sector: {sector}')
                continue

        return models_dict

    def send_order_log_to_mq(self, log: list):
        self.__rabbit.send_message(message=log,
                                   routing_key=ORDER_RELATED_DATA)

    def run_inference(self, l1_dict: dict) -> list:
        orders_list = []
        print(f'l1_dict: {l1_dict}')
        return orders_list


# trader = Trader()
