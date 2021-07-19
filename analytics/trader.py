import numpy as np
import pickle
import os
import emulator.messages as messages

GOT_IT_RESPONSE = {'action': 'gotIt'}
INIT_PCT = -1e3

def current_percentage(l1_dict: dict) -> float:
    try:
        pctBidNet = l1_dict['pctBidNet']
        pctAskNet = l1_dict['pctAskNet']
        if pctBidNet >= 0:
            return pctBidNet
        elif pctAskNet <= 0:
            return pctAskNet
        else:
            return 0
    except TypeError:
        pass


class Trader:
    def __init__(self):
        self.__stocks_l1 = self.__init_indicators()
        self.__factors = self.__init_factors()
        self.__models = self.__init_models()
        self.__positions = {}

    def __init_indicators(self) -> dict:
        print('Inializing indicators...')
        with open(f'analytics/modeling/'
                  f'all_indicators.pkl', 'rb') as i:
            indicators_list = pickle.load(i)

        # Update to get from subscription response or request directly
        indicators_dict = {indicator: {'pctBidNet': INIT_PCT,
                                       'pctAskNet': INIT_PCT}
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
        pct_bid_net = msg['pctBidNet']
        pct_ask_net = msg['pctAskNet']
        bid_l1 = msg['bidL1']
        ask_l1 = msg['askL1']

        # Error
        l1_dict = self.__stocks_l1.get(symbol)
        if not l1_dict:
            self.__stocks_l1[symbol] = {
                'pctBidNet': pct_bid_net,
                'pctAskNet': pct_ask_net
            }
        else:
            self.__stocks_l1[symbol]['pctBidNet'] = pct_bid_net
            self.__stocks_l1[symbol]['pctAskNet'] = pct_ask_net

        # Make trade decision
        try:
            print()
            model_dict = self.__models[symbol]
            # print(self.__models)
            model = model_dict['model']
            # Get indicators
            indicators = self.__factors[symbol]
            # Get indicators l1
            factors_l1 = list(
                map(lambda x: current_percentage(
                    self.__stocks_l1.get(x)), indicators))
            if not INIT_PCT in factors_l1:
                pred_array = np.array(factors_l1).reshape(1, -1)
                prediction = model.predict(pred_array)
                print(f'Symbol Prediction: {prediction}')
                return self.__make_response(msg=msg,
                                            std_err=model_dict['mae'],
                                            pct_bid_net=pct_bid_net,
                                            pct_ask_net=pct_ask_net,
                                            bid_l1=bid_l1,
                                            ask_l1=ask_l1,
                                            prediction=prediction)
            else:
                raise TypeError('One of indicators is not populated yet')
        except KeyError as e:
            print(e)
            print(f'Failed to make inference on message: {msg}')
            return GOT_IT_RESPONSE
        except TypeError as e:
            print(e)

        return GOT_IT_RESPONSE

    def __make_response(self, msg: dict,
                        std_err: float,
                        pct_bid_net: float,
                        pct_ask_net: float,
                        bid_l1: float,
                        ask_l1: float,
                        prediction: float) -> dict:
        if (prediction - pct_ask_net) >= std_err:
            order = messages.order_request()
            order['symbol'] = msg['symbol']
            order['price'] = ask_l1
            order['side'] = 'B'
            return order
        elif (prediction - pct_bid_net) <= -std_err:
            order = messages.order_request()
            order['symbol'] = msg['symbol']
            order['price'] = bid_l1
            order['side'] = 'S'
            return order

        return GOT_IT_RESPONSE
