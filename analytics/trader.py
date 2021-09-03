import numpy as np
import pickle
import os
import copy
from typing import Tuple
import uuid
from pytz import timezone
from datetime import datetime, timedelta
import json
import config.messages as messages
from config.constants import *
from messaging.rabbit_sender import RabbitSender
from messaging.redis_connector import RedisConnector
import redis
import random
import traceback
from analytics.base_trader import BaseTrader
from analytics.app_trader import AppTrader
from analytics.banks_trader import BanksTrader
from analytics.oil_trader import OilTrader
from analytics.semi_trader import SemiTrader
from analytics.renewable_trader import RenewableTrader
from analytics.china_trader import ChinaTrader

sum_update_l1_speed = 0
sum_predict_speed = 0
sum_process_speed = 0

count_update_l1_speed = 0
count_predict_speed = 0
count_process_speed = 0

average_update_l1_speed = 0
average_predict_speed = 0
average_process_speed = 0

EST = timezone('EST')
now_init = datetime.now()
final_dt = datetime(year=now_init.year,
                    month=now_init.month,
                    day=now_init.day,
                    hour=9,
                    minute=27,
                    second=41)


def generate_id() -> str:
    unique_id = uuid.uuid4().__str__().strip()
    unique_id = unique_id.replace('-', '')
    return unique_id


def generate_cid() -> int:
    return random.getrandbits(64)


def current_percentage(l1_dict: dict) -> float:
    try:
        pct_bid_net = l1_dict[PCT_BID_NET]
        pct_ask_net = l1_dict[PCT_ASK_NET]
        if pct_bid_net >= 0:
            return pct_bid_net
        elif pct_ask_net <= 0:
            return pct_ask_net
        else:
            return 0
    except TypeError:
        pass


def get_tickers() -> list:
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
            message = f'Get tickers error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to init factors')
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


def init_stocks_data() -> Tuple[dict, list]:
    print('Inializing indicators...')
    with open(f'analytics/modeling/'
              f'all_indicators.pkl', 'rb') as i:
        indicators_list = pickle.load(i)

    # Update to get from subscription response or request directly
    indicators_dict = {indicator: {PCT_BID_NET: INIT_PCT,
                                   PCT_ASK_NET: INIT_PCT}
                       for indicator in indicators_list}

    print(indicators_dict)

    return indicators_dict, indicators_list


def get_sector_trader(sector: str) -> BaseTrader:
    if sector == APPLICATION_SOFTWARE:
        return AppTrader()
    elif sector == BANKS:
        return BanksTrader()
    elif sector == OIL:
        return OilTrader()
    elif sector == SEMICONDUCTORS:
        return SemiTrader()
    elif sector == RENEWABLE_ENERGY:
        return RenewableTrader()
    elif sector == CHINA:
        return ChinaTrader()
    else:
        raise KeyError(f'Sector trader for {sector} not found')


def init_models() -> dict:
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
            message = f'Init models error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to init models')
            continue

    return models_dict


def init_factors() -> Tuple[dict, dict, dict, dict]:
    print('Initializing factors...')
    sectors_path = 'analytics/modeling/sectors'
    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]

    factors_dict = {}
    stock_to_sector = {}
    sector_to_indicators = {}
    sector_to_trader = {}
    for sector_dir in sectors_dirs:
        sector = sector_dir.split('/')[-1]
        print(f'Sector: {sector}...')
        tickers_indicators_filtered_path \
            = f'{sector_dir}/models/tickers_indicators_filtered.pkl'
        try:
            with open(tickers_indicators_filtered_path, 'rb') as i:
                tickers_indicators_filtered = pickle.load(i)
            factors_dict.update(tickers_indicators_filtered)
            current_sector_stocks = list(tickers_indicators_filtered.keys())
            key = current_sector_stocks[0]
            current_sector_indicators = tickers_indicators_filtered[key]
            sector_to_indicators[sector] = current_sector_indicators
            sector_to_trader[sector] = get_sector_trader(sector)
            for stock in current_sector_stocks:
                stock_to_sector[stock] = sector
            print(f'Sector: {sector} loaded.')
            print(f'-------')
        except Exception as e:
            message = f'Init factors error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to init factors')
            continue
        except KeyError as e:
            message = f'Init factors KeyError: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to init factors')

    return factors_dict, stock_to_sector, sector_to_indicators, sector_to_trader


class Trader:
    def __init__(self):
        self.__rabbit_sender = RabbitSender(RABBIT_MQ_HOST, RABBIT_MQ_PORT)
        self.__redis_connector = RedisConnector()
        self.__tickers = get_tickers()
        self.__stocks_l1, self.__all_indicators = init_stocks_data()
        self.__factors, \
            self.__stock_to_sector,\
            self.__sector_to_indicators, \
            self.__sector_to_trader = init_factors()
        self.__init_policy()
        self.__models = init_models()
        self.__positions = {}
        self.__sent_orders_by_ticker = {}

    def get_subscription_list(self) -> list:
        return self.__tickers

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
        traidable_list = []
        while data:
            symbol_dict = data.pop()
            self.__update_l1(symbol_dict)
            if symbol_dict[SYMBOL] not in self.__all_indicators:
                traidable_list = traidable_list + [symbol_dict]
                print(f'Added {symbol_dict[SYMBOL]} to '
                      f'further processing')
        finish_update = datetime.now()
        delta_update = (finish_update - start).microseconds
        sum_update_l1_speed = sum_update_l1_speed + delta_update
        count_update_l1_speed = count_update_l1_speed + 1
        average_update_l1_speed = sum_update_l1_speed / count_update_l1_speed
        print(f'Update l1 time: {delta_update} microseconds')

        start_predict = datetime.now()
        if traidable_list:
            for symbol_dict in traidable_list:
                try:
                    order = self.__process_symbol_dict(symbol_dict)
                    if order:
                        orders.append(order)
                except Exception as e:
                    message = f'Process l1 message error: ' \
                              f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                    print(message)
                    print(traceback.format_exc())
                    print(f'Failed to process l1 message: {msg_dict}')
                    pass

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
        try:
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
        except KeyError as e:
            message = f'Update l1 KeyError: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to make inference on symbol message: {symbol_dict}')
        except Exception as e:
            message = f'Update l1 error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to make inference on symbol message: {symbol_dict}')

    def send_order_log_to_mq(self, log: list):
        self.__rabbit_sender.send_message(message=log,
                                          routing_key=ORDER_RELATED_DATA)

    def send_market_data_to_mq(self, log: list):
        self.__rabbit_sender.send_message(message=log,
                                          routing_key=MARKET_DATA_TYPE)

    def __init_policy(self):
        traidable_stocks = list(self.__stock_to_sector.keys())
        black_list = ['CS',
                      'ADSK',
                      'REAL',
                      'TUYA',
                      'DOCU',
                      'VIPS',
                      ]  # Add untraidable stocks here
        policy_dict = {APPLICATION_SOFTWARE: NEUTRAL,
                       BANKS: NEUTRAL,
                       OIL: NEUTRAL,
                       RENEWABLE_ENERGY: NEUTRAL,
                       SEMICONDUCTORS: NEUTRAL,
                       CHINA: NEUTRAL}
        delta_dict = {NEUTRAL: {LONG_COEF: 1,
                                SHORT_COEF: 1},
                      BULL: {LONG_COEF: 1/2,
                             SHORT_COEF: 2},
                      BEAR: {LONG_COEF: 2,
                             SHORT_COEF: 1/2},
                      AGG_BULL: {LONG_COEF: 1/4,
                                 SHORT_COEF: 4},
                      AGG_BEAR: {LONG_COEF: 4,
                                 SHORT_COEF: 1/4}}
        acc_info_dict = {BP_KEY: INIT_BP,
                         BP_USAGE_PCT_KEY: BP_USAGE_PCT}
        stock_to_tier_proportion = {}
        for traidable_stock in traidable_stocks:
            if traidable_stock not in black_list:
                print(f'Setting default risk for symbol {traidable_stock}')
                stock_to_tier_proportion[traidable_stock] = 1
            else:
                print(f'Putting symbol {traidable_stock} in blacklist')
                stock_to_tier_proportion[traidable_stock] = 0
        # Map to str
        for key in delta_dict.keys():
            delta_dict[key] = json.dumps(delta_dict[key])
        self.__redis_connector.set_dict(name=POLICY, d=policy_dict)
        self.__redis_connector.set_dict(name=DELTA_COEF, d=delta_dict)
        self.__redis_connector.set_dict(name=STOCK_TO_TIER_PROPORTION,
                                        d=stock_to_tier_proportion)
        self.__redis_connector.set_dict(name=ACCOUNT_INFORMATION,
                                        d=acc_info_dict)
        print(f'Policy inserted')

    def __get_policy(self, sector: str) -> str:
        policy = self.__redis_connector.hm_get(h=POLICY, key=sector)[0]
        print(policy)
        if not policy:
            return NEUTRAL
        return policy

    def __get_deltas(self, policy: str) -> Tuple[float, float]:
        deltas = json.loads(self.__redis_connector.hm_get(h=DELTA_COEF, key=policy)[0])
        if not deltas:
            return 1.0, 1.0
        return float(deltas[LONG_COEF]), float(deltas[SHORT_COEF])

    def __get_tier_prop(self, stock: str) -> float:
        prop = float(self.__redis_connector.hm_get(h=STOCK_TO_TIER_PROPORTION,
                                                   key=stock)[0])
        print(f'For symbol: {stock} prop: {prop}')
        return prop

    def __get_acc_info(self) -> dict:
        acc_info = self.__redis_connector.h_getall(h=ACCOUNT_INFORMATION)
        return acc_info

    def __validate_tier(self, symbol: str) -> bool:
        num_orders_sent = self.__sent_orders_by_ticker.get(symbol)
        cur_time = datetime.now(EST) + timedelta(hours=1)
        cur_time_hour = cur_time.hour
        cur_time_minute = cur_time.minute
        cur_time_second = cur_time.second
        all_invalid_flag = datetime.now() > final_dt
        print('======================================')
        print(f'Validity flag: {all_invalid_flag}')
        print('======================================')
        if all_invalid_flag:  # change to negation
            print('======================================')
            print(f'Validity flag: {all_invalid_flag}')
            print('======================================')
            if not num_orders_sent:
                self.__sent_orders_by_ticker[symbol] = 1
                return True
            if num_orders_sent < 8:
                if cur_time_hour < 8 and num_orders_sent < 2:
                    return True
                elif cur_time_hour == 8 and num_orders_sent < 4:
                    return True
                elif cur_time_hour == 8 and cur_time_minute > 30 and num_orders_sent < 6:
                    return True
                elif cur_time_hour == 9 and num_orders_sent < 8:
                    return True
        return False

    def __process_symbol_dict(self, symbol_dict: dict) -> dict:
        symbol = symbol_dict[SYMBOL]
        symbol_prop = self.__get_tier_prop(stock=symbol)
        # If the tier proportion is 0 (stock is in black list)
        if float(symbol_prop):
            print(f'Valid symbol: {symbol}')
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
                print(f'{symbol} indicators: ')
                print(indicators)
                # Get indicators l1
                factors_l1 = list(
                    map(lambda x: current_percentage(
                        self.__stocks_l1.get(x)), indicators))

                print(f'{symbol} current factors:')
                print(factors_l1)
                if INIT_PCT not in factors_l1:
                    valid_tier = self.__validate_tier(symbol=symbol)
                    if valid_tier:
                        pred_array = np.array(factors_l1).reshape(1, -1)
                        prediction = model.predict(pred_array)[0]
                        print(f'{symbol} prediction: {prediction}\n'
                              f'current pctBidNet: {pct_bid_net}, '
                              f'current pctAskNet: {pct_ask_net}')
                        std_err = model_dict['mae']

                        # Check for trade opportunity
                        symbol_sector = self.__stock_to_sector[symbol]
                        symbol_trader = self.__sector_to_trader[symbol_sector]
                        symbol_policy = self.__get_policy(sector=symbol_sector)
                        delta_long_coef, delta_short_coef = self.__get_deltas(policy=symbol_policy)
                        acc_info = self.__get_acc_info()
                        bp = float(acc_info[BP_KEY])
                        bp_usage_pct = float(acc_info[BP_USAGE_PCT_KEY])
                        print(f'BP: {bp}')
                        print(f'bp type: {type(bp)}')
                        print(f'BP_USAGE_PCT: {bp_usage_pct}')
                        print(f'bp_usage_pct type: {type(bp_usage_pct)}')
                        order_data = symbol_trader.get_order(prediction=prediction,
                                                             pct_bid_net=pct_bid_net,
                                                             pct_ask_net=pct_ask_net,
                                                             indicators=indicators,
                                                             factors_l1=factors_l1,
                                                             close=close,
                                                             symbol=symbol,
                                                             bid_l1=bid_l1,
                                                             ask_l1=ask_l1,
                                                             bid_venue=bid_venue,
                                                             ask_venue=ask_venue,
                                                             std_err=std_err,
                                                             policy=symbol_policy,
                                                             prop=symbol_prop,
                                                             delta_long_coef=delta_long_coef,
                                                             delta_short_coef=delta_short_coef,
                                                             bp=bp,
                                                             bp_usage_pct=bp_usage_pct)
                        self.__sent_orders_by_ticker[symbol] += 1
                        return order_data

                else:
                    raise TypeError('One of indicators is not populated yet')

            except KeyError as e:
                message = f'Process symbol dict error: ' \
                          f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                print(message)
                print(traceback.format_exc())
                print(f'Failed to make inference on symbol message: {symbol_dict}')

            except TypeError as e:
                message = f'Process symbol dict error: ' \
                          f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                print(message)
                print(traceback.format_exc())
                print(f'Failed to make inference on symbol message: {symbol_dict}')

            except Exception as e:
                message = f'Process symbol dict error: ' \
                          f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                print(message)
                print(traceback.format_exc())
                print(f'Failed to make inference on symbol message: {symbol_dict}')

        return {}


# trader = Trader()
# print(trader.get_subscription_list())
# import pandas as pd
# pd.DataFrame(trader.get_subscription_list()).to_csv('analytics/modeling/tickers.csv')
