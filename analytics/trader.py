import numpy as np
import pickle
import os
import copy
from typing import Tuple
import uuid
from datetime import datetime
import config.messages as messages
from config.constants import *
from messaging.rabbit_sender import RabbitSender
import random

sum_update_l1_speed = 0
sum_predict_speed = 0
sum_process_speed = 0

count_update_l1_speed = 0
count_predict_speed = 0
count_process_speed = 0

average_update_l1_speed = 0
average_predict_speed = 0
average_process_speed = 0


def generate_id() -> str:
    unique_id = uuid.uuid4().__str__().strip()
    unique_id = unique_id.replace('-', '')
    return unique_id


def generate_cid() -> int:
    return random.getrandbits(64)


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
        self.__sent_orders_by_ticker = {}
        self.__rabbit_sender = RabbitSender(RABBIT_MQ_HOST, RABBIT_MQ_PORT)
        self.__tickers = self.__get_tickers()
        self.__stocks_l1, self.__all_indicators = self.__init_indicators()
        self.__factors = self.__init_factors()
        self.__models = self.__init_models()
        self.__positions = {}

    def get_subscription_list(self) -> list:
        return self.__tickers

    def __get_tickers(self) -> list:
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

        # Getting earnings calendar manually every day
        # https://hosting.briefing.com/cschwab/
        # Calendars/EarningsCalendar5Weeks.htm
        with open('analytics/modeling/reports') as i:
            reports = i.read().splitlines()
            print(f'Reports: {reports}')

        all_tickers =\
            [ticker for ticker in all_tickers if ticker not in reports]

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


    def process_l1_message(self, msg_dict: dict) -> list:
        global average_update_l1_speed, average_predict_speed,\
            average_process_speed, sum_update_l1_speed, sum_predict_speed, \
            sum_process_speed, count_update_l1_speed, count_predict_speed,\
            count_process_speed
        # Update stock's l1
        msg = copy.deepcopy(msg_dict)
        data = msg[DATA]
        orders = []
        start = datetime.now()
        for symbol_dict in data:
            self.__update_l1(symbol_dict)
            if symbol_dict[SYMBOL] in self.__all_indicators:
                data.remove(symbol_dict)
        finish_update = datetime.now()
        delta_update = (finish_update - start).microseconds
        sum_update_l1_speed = sum_update_l1_speed + delta_update
        count_update_l1_speed = count_update_l1_speed + 1
        average_update_l1_speed = sum_update_l1_speed / count_update_l1_speed
        print(f'Update l1 time: {delta_update} microseconds')

        start_predict = datetime.now()
        if data:
            for symbol_dict in data:
                order = self.__process_symbol_dict(symbol_dict)
                if order:
                    orders.append(order)
        finish = datetime.now()
        delta_predict = (finish - start_predict).microseconds
        delta = (finish - start).microseconds
        sum_predict_speed = sum_predict_speed + delta_predict
        count_predict_speed = count_predict_speed + 1
        average_predict_speed = sum_predict_speed / count_predict_speed
        sum_process_speed = sum_process_speed + delta
        count_process_speed = count_process_speed + 1
        average_process_speed = sum_process_speed / count_process_speed
        print(f'Predict time: {delta_predict} microseconds')
        print(f'Process time: {delta} microseconds')

        print(f'Average update l1 time: {average_update_l1_speed} microseconds')
        print(f'Average predict time: {average_predict_speed} microseconds')
        print(f'Average process time: {average_process_speed} microseconds')

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

    def send_order_log_to_mq(self, log: list):
        self.__rabbit_sender.send_message(message=log,
                                          routing_key=ORDER_RELATED_DATA)

    def send_market_data_to_mq(self, log: list):
        self.__rabbit_sender.send_message(message=log,
                                          routing_key=MARKET_DATA_TYPE)

    def __process_symbol_dict(self, symbol_dict: dict) -> dict:
        symbol = symbol_dict[SYMBOL]
        pct_bid_net = symbol_dict[PCT_BID_NET]
        pct_ask_net = symbol_dict[PCT_ASK_NET]
        bid_l1 = symbol_dict[BID]
        ask_l1 = symbol_dict[ASK]
        bid_venue = symbol_dict[BID_VENUE]
        # bid_venue = 1
        ask_venue = symbol_dict[ASK_VENUE]
        # ask_venue = 1
        close = symbol_dict[CLOSE]

        # Make trade decision
        try:
            model_dict = self.__models[symbol]
            model = model_dict[MODEL]
            # Get indicators
            indicators = self.__factors[symbol]
            # Get indicators l1
            factors_l1 = list(
                map(lambda x: current_percentage(
                    self.__stocks_l1.get(x)), indicators))

            print(f'{symbol} current factors:')
            print(factors_l1)

            if INIT_PCT not in factors_l1 or INIT_PCT in factors_l1:
                pred_array = np.array(factors_l1).reshape(1, -1)
                prediction = model.predict(pred_array)
                print(f'{symbol} prediction: {prediction}\n'
                      f'current pctBidNet: {pct_bid_net}, '
                      f'current pctAskNet: {pct_ask_net}')
                std_err = model_dict['mae']

                # Check for trade opportunity
                order_data = self.__get_order(prediction,
                                              pct_bid_net,
                                              pct_ask_net,
                                              indicators,
                                              factors_l1,
                                              close,
                                              symbol,
                                              bid_l1,
                                              ask_l1,
                                              bid_venue,
                                              ask_venue)

                return order_data

            else:
                raise TypeError('One of indicators is not populated yet')

        except KeyError as e:
            print(e)
            print(f'Failed to make inference on symbol message: {symbol_dict}')

        except TypeError as e:
            print(e)

        except Exception as e:
            print(e)

        return {}

    def __get_order(self,
                    prediction,
                    pct_bid_net,
                    pct_ask_net,
                    indicators,
                    factors_l1,
                    close,
                    symbol,
                    bid_l1,
                    ask_l1,
                    bid_venue,
                    ask_venue) -> dict:
        side_params = {
            # Long params
            BUY: {
                PRICE: ask_l1,
                VENUE: ask_venue,
                PCT_NET: pct_ask_net,
            },
            # Short params
            SELL: {
                PRICE: bid_l1,
                VENUE: bid_venue,
                PCT_NET: pct_bid_net
            }
        }
        order_data = {}
        delta = prediction - pct_ask_net
        trade_flag = delta >= 0 or delta <= 0  # change to std_err and -std_err
        if trade_flag:
            side = BUY if np.sign(delta) > 0 else SELL
            order_params = side_params[side]
            order_related_data_dict = dict(zip(indicators, factors_l1))
            order_data[ORDER_RELATED_DATA] = order_related_data_dict
            target = float(close + float(prediction / 100) * close)
            order = messages.order_request()
            order[ORDER][DATA][SYMBOL] = symbol
            order[ORDER][DATA][PRICE] = order_params[PRICE]
            order[ORDER][DATA][SIDE] = side
            order[ORDER][DATA][SIZE] = 1
            order[ORDER][DATA][VENUE] = order_params[VENUE]
            order[ORDER][DATA][TARGET] = order_params[PRICE] + 0.50  # change to target
            order[ORDER][CID] = generate_cid()
            order_data[ORDER_DATA] = order
            print(f'Stock: {symbol}, {side} '
                  f'{order_params[PRICE]},\n'
                  f'Current ask: {order_params[PCT_NET]}, '
                  f'prediction: {prediction}, '
                  f'target: {target}')

        # Change in future
        if self.__sent_orders_by_ticker.get(symbol):
            return {}
        else:
            self.__sent_orders_by_ticker[symbol] = True
            return order_data
