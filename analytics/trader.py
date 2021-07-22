import numpy as np
import pickle
import os
from typing import Tuple
import emulator.messages as messages

INIT_PCT = -1e3
SYMBOL = 'symbol'
ORDER = 'order'
DATA = 'data'
PCT_BID_NET = 'pctBidNet'
PCT_ASK_NET = 'pctAskNet'
BID = 'bid'
ASK = 'ask'
BID_VENUE = 'bidVenue'
ASK_VENUE = 'askVenue'
PRICE = 'limit'
SIDE = 'side'
VENUE = 'venue'


def current_percentage(l1_dict: dict) -> float:
    try:
        pctBidNet = l1_dict[PCT_BID_NET]
        pctAskNet = l1_dict[PCT_ASK_NET]
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
        self.__stocks_l1, self.__all_indicators = self.__init_indicators()
        self.__factors = self.__init_factors()
        self.__models = self.__init_models()
        self.__positions = {}

    def get_subscription_list(self) -> list:
        sectors_path = 'analytics/modeling/sectors'
        sectors_dirs = \
            [f.path for f in os.scandir(sectors_path) if f.is_dir()]

        all_tickers = []

        for sector_dir in sectors_dirs:
            sector = sector_dir.split('/')[-1]
            if 'tickers' not in os.listdir(sector_dir):
                print(f'tickers dir missing '
                      f'for sector: {sector}')
                continue

            tickers_files = os.listdir(f'{sector_dir}/tickers')

            try:
                for f in tickers_files:
                    with open(f'{sector_dir}/tickers/{f}', 'rb') as inp:
                        tickers = pickle.load(inp)
                        all_tickers = all_tickers + tickers

            except Exception as e:
                print(e)
                print(f'Failed to load tickers for sector: {sector}')
                continue

        all_tickers = list(set(all_tickers))

        return all_tickers

    def __init_indicators(self) -> Tuple[dict, list]:
        print('Inializing indicators...')
        with open(f'analytics/modeling/'
                  f'all_indicators.pkl', 'rb') as i:
            indicators_list = pickle.load(i)

        # Update to get from subscription response or request directly
        indicators_dict = {indicator: {PCT_BID_NET: INIT_PCT,
                                       PCT_ASK_NET: INIT_PCT}
                           for indicator in indicators_list}

        return indicators_dict, indicators_list

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

    def process_l1_message(self, msg: dict) -> list:
        # Update stock's l1
        data = msg[DATA]
        orders = []
        for symbol_dict in data:
            self.__update_l1(symbol_dict)
            if symbol_dict[SYMBOL] in self.__all_indicators:
                data.remove(symbol_dict)

        if data:
            for symbol_dict in data:
                order = self.__make_trade_decision(symbol_dict)
                if order:
                    orders.append(order)

        return orders

    def __update_l1(self, symbol_dict: dict):
        symbol = symbol_dict[SYMBOL]
        pct_bid_net = symbol_dict[PCT_BID_NET]
        pct_ask_net = symbol_dict[PCT_ASK_NET]
        l1_dict = self.__stocks_l1.get(symbol)
        if not l1_dict:
            self.__stocks_l1[symbol] = {
                PCT_BID_NET: pct_bid_net,
                PCT_ASK_NET: pct_ask_net
            }
        else:
            self.__stocks_l1[symbol][PCT_BID_NET] = pct_bid_net
            self.__stocks_l1[symbol][PCT_ASK_NET] = pct_ask_net

    def __make_trade_decision(self, symbol_dict: dict) -> dict:
        symbol = symbol_dict[SYMBOL]
        pct_bid_net = symbol_dict[PCT_BID_NET]
        pct_ask_net = symbol_dict[PCT_ASK_NET]
        bid_l1 = symbol_dict[BID]
        ask_l1 = symbol_dict[ASK]
        bid_venue = symbol_dict[BID_VENUE]
        ask_venue = symbol_dict[ASK_VENUE]

        # Make trade decision
        try:
            model_dict = self.__models[symbol]
            model = model_dict['model']
            # Get indicators
            indicators = self.__factors[symbol]
            # Get indicators l1
            factors_l1 = list(
                map(lambda x: current_percentage(
                    self.__stocks_l1.get(x)), indicators))

            print(f'{symbol} current factors:')
            print(factors_l1)

            if not INIT_PCT in factors_l1:
                pred_array = np.array(factors_l1).reshape(1, -1)
                prediction = model.predict(pred_array)
                print(f'{symbol} prediction: {prediction}\n'
                      f'current pctBidNet: {pct_bid_net}, '
                      f'current pctAskNet: {pct_ask_net}')
                std_err = model_dict['mae']

                # Check for long opportunity
                if (prediction - pct_ask_net) >= std_err:
                    order = messages.order_request()
                    order[ORDER][DATA][SYMBOL] = symbol
                    order[ORDER][DATA][PRICE] = ask_l1
                    order[ORDER][DATA][SIDE] = 'B'
                    order[ORDER][DATA][VENUE] = ask_venue
                    print(f'Stock: {symbol}, LONG {ask_l1},\n'
                          f'Current ask: {pct_ask_net}, '
                          f'prediction: {prediction}')
                    return order

                # Check for short opportunity
                elif (prediction - pct_bid_net) <= -std_err:
                    order = messages.order_request()
                    order[ORDER][DATA][SYMBOL] = symbol
                    order[ORDER][DATA][PRICE] = bid_l1
                    order[ORDER][DATA][SIDE] = 'S'
                    order[ORDER][DATA][VENUE] = bid_venue
                    print(f'Stock: {symbol}, SHORT {bid_l1},\n'
                          f'Current bid: {pct_bid_net}, '
                          f'prediction: {prediction}')
                    return order
            else:
                raise TypeError('One of indicators is not populated yet')

        except KeyError as e:
            print(e)
            print(f'Failed to make inference on symbol message: {symbol_dict}')
            return {}

        except TypeError as e:
            print(e)

        return {}
