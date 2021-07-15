import pickle
import os
import emulator.messages as messages

GOT_IT_RESPONSE = {'action': 'gotIt'}

class Trader:
    def __init__(self):
        self.__stocks_l1 = self.__init_indicators()
        self.__factors = self.__init_factors()
        self.__models = self.__init_models()
        self.__positions = {}

    def __init_indicators(self) -> dict:
        print('Inializing indicators...')
        with open(f'analytics/modeling/sectors/'
                  f'all_indicators.pkl', 'rb') as i:
            indicators_list = pickle.load(i)

        # Update to get from subscription response or request directly
        indicators_dict = {indicator: {'%bidNet': 0, '%askNet': 0}
                           for indicator in indicators_list}

        return indicators_dict

    def __init_factors(self) -> dict:
        print('Initializing factors...')
        sectors_path = 'analytics/modeling/sectors'
        sectors_dirs = \
            [f.path for f in os.scandir(sectors_path) if f.is_dir()]

        factors_dict = {}
        for sector_dir in sectors_dirs:
            sector = sector_dir.split('/')[-1]
            print(f'Sector: {sector}...')
            tickers_indicators_filtered_path\
                = f'{sector_dir}/models/tickers_indicators_filtered.pkl'
            try:
                with open(tickers_indicators_filtered_path, 'rb') as i:
                    tickers_indicators_filtered = pickle.load(i)
                factors_dict.update(tickers_indicators_filtered)
                print(f'Sector: {sector} loaded.')
                print(f'-------')
            except Exception as e:
                print(e)
                print(f'Failed to load tickers for sector: {sector}')
                continue

        return factors_dict

    def __init_models(self) -> dict:
        print('Initializing models...')
        sectors_path = 'analytics/modeling/sectors'
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

    def process_stock_l1_message(self, msg: dict) -> dict:
        # Update stock's l1
        symbol = msg['symbol']
        bid_net = msg['%bidNet']
        ask_net = msg['%askNet']

        self.__stocks_l1[symbol]['%bidNet'] = bid_net
        self.__stocks_l1[symbol]['%askNet'] = ask_net

        # Make trade decision
        try:

            model_dict = self.__models.get(symbol)
            model = model_dict['model']
            # Get indicators
            indicators = self.__factors.get(symbol)
            # Get indicators l1
            factors_l1 = list(map(self.__stocks_l1.get, indicators))
            # factors_array = get_factors_array()
        except Exception as e:
            print(e)
            print(f"Failed to make inference on message: {msg}")
            return GOT_IT_RESPONSE

        response_dict = self.__make_order(msg=msg)
        return response_dict

    def __make_order(self, msg: dict) -> dict:
        order = messages.order_request()
        order['symbol'] = msg['symbol']
        return order


trader = Trader()
