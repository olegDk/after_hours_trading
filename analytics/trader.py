import numpy as np
import pickle
import os
from typing import Tuple
import uuid
from pytz import timezone
from datetime import datetime, timedelta
import json
import config.messages as messages
from config.constants import *
from messaging.rabbit_sender import RabbitSender
from messaging.redis_connector import RedisConnector
from news_analyzer.news_analyzer import NewsAnalyzer
import random
import traceback

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
                    second=0,
                    tzinfo=EST)


def generate_id() -> str:
    unique_id = uuid.uuid4().__str__().strip()
    unique_id = unique_id.replace('-', '')
    return unique_id


def generate_cid() -> int:
    return random.getrandbits(64)


def get_position_size(price: float,
                      bp: float,
                      bp_usage_pct: float,
                      prop: float
                      ) -> int:
    size = int((bp * bp_usage_pct * 0.1 * 0.0625 * prop) / (price + 1e-7))
    if size < 5:
        size = 5
    elif 95 <= size < 100:
        size = 100
    return size


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


def invert_dict(d: dict) -> dict:
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                inverse[item].append(key)
    return inverse


def extend_dict(fr: dict, to: dict) -> dict:
    extended = {}
    for key_1 in fr:
        key_1_vals = []
        for key_2 in fr[key_1]:
            key_1_vals = key_1_vals + to[key_2]
        extended[key_1] = list(set(key_1_vals))
    return extended


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

    all_tickers = \
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


def init_factors() -> Tuple[dict, dict, dict, dict, dict, dict]:
    print('Initializing factors...')
    sectors_path = 'analytics/modeling/sectors'
    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]

    factors_dict = {}
    main_etfs_dict = {}
    stock_to_sector = {}
    sector_to_indicators = {}
    sector_to_stocks = {}

    for sector_dir in sectors_dirs:
        sector = sector_dir.split('/')[-1]
        print(f'Sector: {sector}...')
        tickers_indicators_filtered_path \
            = f'{sector_dir}/models/tickers_indicators_filtered.pkl'
        main_etfs_filtered_path = f'{sector_dir}/models/tickers_main_etf_filtered.pkl'
        try:
            with open(tickers_indicators_filtered_path, 'rb') as i:
                tickers_indicators_filtered = pickle.load(i)
            factors_dict.update(tickers_indicators_filtered)
            with open(main_etfs_filtered_path, 'rb') as i:
                main_etfs_filtered = pickle.load(i)
            main_etfs_dict.update(main_etfs_filtered)
            current_sector_stocks = list(tickers_indicators_filtered.keys())
            key = current_sector_stocks[0]
            current_sector_indicators = tickers_indicators_filtered[key]
            sector_to_indicators[sector] = current_sector_indicators
            sector_to_stocks[sector] = current_sector_stocks
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

    indicator_to_sectors = invert_dict(d=sector_to_indicators)
    indicator_to_stocks = extend_dict(fr=indicator_to_sectors, to=sector_to_stocks)

    return factors_dict, main_etfs_dict, stock_to_sector, \
           sector_to_indicators, indicator_to_sectors, indicator_to_stocks


def adjust_limit_price(side,
                       l1_price,
                       target,
                       prem_low,
                       prem_high,
                       vwap,
                       prediction_main_etf,
                       std_err_main_etf,
                       close) -> float:
    prediction_main_etf_price = float(close + float(prediction_main_etf / 100) * close)
    if vwap:  # and (prem_low <= vwap <= prem_high):  # if there were another trades

        # short logic
        if side == SELL:
            short_bound = prediction_main_etf_price + std_err_main_etf / 3
            if prem_low <= l1_price <= prem_high:
                if l1_price >= vwap:
                    return max([l1_price, short_bound])
                else:
                    adjusted_price = round(vwap - float(np.abs(vwap-target)) / 3, 2)
                    return max([adjusted_price, l1_price, short_bound])
            elif l1_price >= prem_high:
                return max([l1_price, short_bound])
            elif l1_price <= prem_low:
                adjusted_price = round(vwap - float(np.abs(vwap-target)) / 3, 2)
                return max([adjusted_price, l1_price, short_bound])

        # long logic
        elif side == BUY:
            long_bound = prediction_main_etf_price - std_err_main_etf / 3
            if prem_low <= l1_price <= prem_high:
                if l1_price <= vwap:
                    return min([l1_price, long_bound])
                else:
                    adjusted_price = round(vwap + float(np.abs(vwap-target)) / 3, 2)
                    return min([adjusted_price, l1_price, long_bound])
            elif l1_price <= prem_low:
                return min([l1_price, long_bound])
            elif l1_price >= prem_high:
                adjusted_price = round(vwap + float(np.abs(vwap-target)) / 3, 2)
                return min([adjusted_price, l1_price, long_bound])

    return l1_price


def get_order(prediction,
              prediction_main_etf,
              pct_bid_net,
              pct_ask_net,
              indicators,
              factors_l1,
              close,
              symbol,
              bid_l1,
              ask_l1,
              bid_venue,
              ask_venue,
              vwap,
              prem_high,
              prem_low,
              imb,
              vol,
              std_err,
              std_err_main_etf,
              lower_sigma,
              upper_sigma,
              lower_two_sigma,
              upper_two_sigma,
              policy,
              prop,
              delta_long_coef,
              delta_short_coef,
              bp,
              bp_usage_pct) -> dict:
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
    delta_long = prediction - pct_ask_net
    delta_short = prediction - pct_bid_net
    trade_flag = delta_long >= std_err * delta_long_coef or \
                 delta_short <= -std_err * delta_short_coef
    if trade_flag:
        side = BUY if np.sign(delta_long) > 0 else SELL
        order_params = side_params[side]
        order_related_data_dict = dict(zip(indicators, factors_l1))
        order_related_data_dict.update({POLICY: policy,
                                        LONG_COEF: delta_long_coef,
                                        SHORT_COEF: delta_short_coef,
                                        PREDICTION_KEY: prediction})
        order_data[ORDER_RELATED_DATA] = order_related_data_dict
        target = float(close + float(prediction / 100) * close)
        order = messages.order_request()
        order[ORDER][DATA][SYMBOL] = symbol
        price = order_params[PRICE]
        order[ORDER][DATA][LIMIT] = adjust_limit_price(side=side,
                                                       l1_price=price,
                                                       target=target,
                                                       prem_low=prem_low,
                                                       prem_high=prem_high,
                                                       vwap=vwap,
                                                       prediction_main_etf=prediction_main_etf,
                                                       std_err_main_etf=std_err_main_etf,
                                                       close=close
                                                       )
        order[ORDER][DATA][SIDE] = side
        order[ORDER][DATA][SIZE] = get_position_size(price=order_params[PRICE],
                                                     bp=bp,
                                                     bp_usage_pct=bp_usage_pct,
                                                     prop=prop)
        order[ORDER][DATA][VENUE] = order_params[VENUE]
        order[ORDER][DATA][TARGET] = target
        order[ORDER][CID] = generate_cid()
        order_data[ORDER_DATA] = order

    return order_data


class Trader:
    def __init__(self):
        self.__rabbit_sender = RabbitSender(RABBIT_MQ_HOST, RABBIT_MQ_PORT)
        self.__redis_connector = RedisConnector(REDIS_HOST, REDIS_PORT)
        self.__tickers = get_tickers()
        self.__stocks_l1, self.__all_indicators = init_stocks_data()
        self.__factors, \
        self.__main_etfs, \
        self.__stock_to_sector, \
        self.__sector_to_indicators, \
        self.__indicator_to_sectors, \
        self.__indicators_to_stocks = init_factors()
        self.__init_policy()
        self.__models = init_models()
        self.positions = {}
        self.__orders = {}
        self.__sent_orders_by_ticker = {}
        self.__na = NewsAnalyzer()

    def get_subscription_list(self) -> list:
        return self.__tickers

    def update_account_information(self,
                                   acc_info: dict):
        self.__redis_connector.h_del(h=POSITIONS)
        if acc_info:
            bp_info = self.__get_acc_info()
            bp = float(bp_info[BP_KEY])
            bp_usage_pct = float(bp_info[BP_USAGE_PCT_KEY])
            cur_bp_per_tier = bp * bp_usage_pct * 0.1 / MAX_ORDERS
            positions = acc_info.get(POSITIONS)
            if positions:
                for pos_dict in positions:
                    symbol = pos_dict.get(SYMBOL)
                    pos_size = pos_dict[SIZE]
                    pos_side = pos_dict[SIDE]
                    pos_price = pos_dict[PRICE]
                    pos_investment = pos_size * pos_price
                    if symbol and symbol in self.positions:
                        self.positions[symbol][SIZE] = pos_size
                        self.positions[symbol][SIDE] = pos_side
                        self.positions[symbol][PRICE] = pos_price
                        self.positions[symbol][INVESTMENT] = pos_investment
                    elif symbol:
                        self.positions[symbol] = {
                            SIZE: pos_size,
                            SIDE: pos_side,
                            PRICE: pos_price,
                            INVESTMENT: pos_investment
                        }
                positions_to_save = {}
                for key in self.positions.keys():
                    positions_to_save[key] = json.dumps(self.positions[key])
                self.__redis_connector.set_dict(name=POSITIONS,
                                                d=positions_to_save,
                                                rewrite=True)

            orders = acc_info.get(ORDERS)
            if orders:
                for order_dict in orders:
                    symbol = order_dict.get(SYMBOL)
                    if symbol and symbol in self.__orders:
                        self.__orders[symbol] = self.__orders[symbol] + [order_dict]
                    elif symbol:
                        self.__orders[symbol] = [order_dict]
                orders_to_save = {}
                for key in self.__orders.keys():
                    orders_to_save[key] = json.dumps(self.__orders[key])
                self.__redis_connector.set_dict(name=ORDERS,
                                                d=orders_to_save,
                                                rewrite=True)

            # Calculate number of sent tiers by symbol if exist
            if self.positions:
                for symbol in self.positions.keys():
                    bp_symbol = self.positions[symbol][SIZE] * \
                                self.positions[symbol][PRICE]
                    print(f'BP symbol: {symbol}: {bp_symbol}')
                    n_tiers_symbol = int(bp_symbol / cur_bp_per_tier)
                    print(f'cur_bp_per_tier: {cur_bp_per_tier}')
                    print(f'n_tiers_symbol symbol: {symbol}: {n_tiers_symbol}')
                    if symbol in self.__sent_orders_by_ticker:
                        self.__sent_orders_by_ticker[symbol] = \
                            self.__sent_orders_by_ticker[symbol] + n_tiers_symbol
                    else:
                        self.__sent_orders_by_ticker[symbol] = n_tiers_symbol
            if self.__orders:
                for symbol in self.__orders:
                    n_tiers_symbol = 0
                    side = None
                    if symbol in self.positions:
                        side = self.positions[symbol][SIDE]
                    for order in self.__orders[symbol]:
                        order_side = order[SIDE]
                        if not side or side == order_side:
                            bp_order = order[SIZE] * order[PRICE]
                            n_tiers_order = int(bp_order / cur_bp_per_tier)
                            n_tiers_symbol += n_tiers_order
                            print(f'BP symbol order: {symbol}: {bp_order}')
                            print(f'cur_bp_per_tier: {cur_bp_per_tier}')
                            print(f'n_tiers_symbol symbol: {symbol}: {n_tiers_symbol}')
                    if symbol in self.__sent_orders_by_ticker:
                        self.__sent_orders_by_ticker[symbol] = \
                            self.__sent_orders_by_ticker[symbol] + n_tiers_symbol
                    else:
                        self.__sent_orders_by_ticker[symbol] = n_tiers_symbol
            if self.__sent_orders_by_ticker:
                self.__redis_connector.set_dict(name=SENT_ORDERS_BY_TICKER,
                                                d=self.__sent_orders_by_ticker,
                                                rewrite=True)

    def process_md_message(self, msg_dict: dict) -> list:
        global average_update_l1_speed, average_predict_speed, \
            average_process_speed, sum_update_l1_speed, sum_predict_speed, \
            sum_process_speed, count_update_l1_speed, count_predict_speed, \
            count_process_speed
        # Update stock's l1
        data = msg_dict[DATA].copy()
        orders = []
        start = datetime.now()
        traidable_list = []
        indicators_names_list = []
        while data:
            symbol_dict = data.pop()
            sym = symbol_dict[SYMBOL]
            self.__update_l1(symbol_dict)
            if sym not in self.__all_indicators:
                traidable_list = traidable_list + [symbol_dict]
                print(f'Added {sym} to '
                      f'further processing')
            else:
                indicators_names_list = indicators_names_list + [sym]
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
                    order = self.process_symbol_dict(symbol_dict)
                    if order:
                        orders.append(order)
                except Exception as e:
                    message = f'Process l1 message error: ' \
                              f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                    print(message)
                    print(traceback.format_exc())
                    print(f'Failed to process l1 message')
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
            pass
        except Exception as e:
            message = f'Update l1 error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to make inference on symbol message: {symbol_dict}')
            pass

    def process_order_report(self, msg: dict):
        try:
            symbol = msg[SYMBOL]
            side = msg[SIDE]
            size = int(msg[SIZE])
            price = float(msg[PRICE])
            investment = size * price
            if symbol in self.positions:
                pos_size = self.positions[symbol][SIZE]
                pos_side = self.positions[symbol][SIDE]
                pos_price = self.positions[symbol][PRICE]
                pos_investment = self.positions[symbol][INVESTMENT]
                # if add
                if side == pos_side:
                    pos_size = pos_size + size
                    self.positions[symbol][SIZE] = pos_size
                    pos_investment = pos_investment + investment
                    self.positions[symbol][INVESTMENT] = pos_investment
                    pos_price = pos_investment / pos_size
                    self.positions[symbol][PRICE] = pos_price
                # if cover
                elif size <= pos_size:
                    pos_size = pos_size - size
                    self.positions[symbol][SIZE] = pos_size
                    self.positions[symbol][INVESTMENT] = pos_size * pos_price
                # if flip
                elif size > pos_size:
                    pos_size = int(np.abs(pos_size - size))
                    self.positions[symbol][SIZE] = pos_size
                    self.positions[symbol][SIDE] = side
                    self.positions[symbol][PRICE] = price
                    self.positions[symbol][INVESTMENT] = pos_size * price
            # If there is no position for current symbol
            else:
                self.positions[symbol] = {
                    SIZE: size,
                    SIDE: side,
                    PRICE: price,
                    INVESTMENT: investment
                }
            # Redis update for current symbol
            print(self.positions[symbol])
            symbol_dict_str = json.dumps(self.positions[symbol])
            self.__redis_connector.h_set_str(h=POSITIONS,
                                             key=symbol,
                                             value=symbol_dict_str)
        except Exception as e:
            message = f'Process order report error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to process order report: {msg}')

    def send_order_log_to_mq(self, log: list):
        self.__rabbit_sender.send_message(message=log,
                                          routing_key=ORDER_RELATED_DATA)

    def send_market_data_to_mq(self, log: list):
        self.__rabbit_sender.send_message(message=log,
                                          routing_key=MARKET_DATA_TYPE)

    def send_news_data_to_mq(self, log: list):
        self.__rabbit_sender.send_message(message=log,
                                          routing_key=NEWS_TYPE)

    def __init_policy(self):
        traidable_stocks = list(self.__stock_to_sector.keys())
        black_list = []  # Add untraidable stocks here
        policy_dict = {APPLICATION_SOFTWARE: NEUTRAL,
                       OIL: NEUTRAL,
                       RENEWABLE_ENERGY: NEUTRAL,
                       SEMICONDUCTORS: NEUTRAL,
                       CHINA: NEUTRAL,
                       GOLD: NEUTRAL,
                       DOW_JONES: NEUTRAL,
                       STEEL: NEUTRAL
                       }
        delta_dict = {NEUTRAL: {LONG_COEF: 1,
                                SHORT_COEF: 1},
                      BULL: {LONG_COEF: 1 / 2,
                             SHORT_COEF: 2},
                      BEAR: {LONG_COEF: 2,
                             SHORT_COEF: 1 / 2},
                      AGG_BULL: {LONG_COEF: 1 / 4,
                                 SHORT_COEF: 4},
                      AGG_BEAR: {LONG_COEF: 4,
                                 SHORT_COEF: 1 / 4}}
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
        if not policy:
            return NEUTRAL
        return policy

    def __get_deltas(self, policy: str) -> Tuple[float, float]:
        deltas = json.loads(self.__redis_connector.hm_get(h=DELTA_COEF, key=policy)[0])
        if not deltas:
            return 1.0, 1.0
        return float(deltas[LONG_COEF]), float(deltas[SHORT_COEF])

    def get_tier_prop(self, symbol: str) -> float:
        prop = self.__redis_connector.hm_get(h=STOCK_TO_TIER_PROPORTION,
                                             key=symbol)[0]
        if prop:
            prop = float(prop)
        else:
            prop = 0.0
        return prop

    def __get_acc_info(self) -> dict:
        acc_info = self.__redis_connector.h_getall(h=ACCOUNT_INFORMATION)
        return acc_info

    def __get_num_orders_sent(self, symbol: str) -> int:
        n = self.__redis_connector.hm_get(h=SENT_ORDERS_BY_TICKER,
                                          key=symbol)[0]
        if n:
            n = int(n)
        else:
            n = 0
        return n

    def validate_tier(self, symbol: str) -> bool:
        # num_orders_sent = self.__sent_orders_by_ticker.get(symbol)
        num_orders_sent = self.__get_num_orders_sent(symbol=symbol)
        cur_time = datetime.now(EST) + timedelta(hours=1)
        cur_time_hour = cur_time.hour
        cur_time_minute = cur_time.minute
        all_invalid_flag = cur_time > final_dt
        if not all_invalid_flag:
            if not num_orders_sent:
                if cur_time_hour < 9:
                    return True
                else:
                    return True
            if num_orders_sent < MAX_ORDERS:
                if cur_time_hour < 8 and \
                        num_orders_sent < BEFORE_8_N_ORDERS:
                    return True
                elif cur_time_hour == 8 and \
                        num_orders_sent < BEFORE_8_30_N_ORDERS:
                    return True
                elif cur_time_hour == 8 and \
                        cur_time_minute > 30 and \
                        num_orders_sent < BEFORE_9_N_ORDERS:
                    return True
                elif cur_time_hour == 9 and \
                        cur_time_minute < 15 and \
                        num_orders_sent < BEFORE_9_15_ORDERS:
                    return True
                elif cur_time_hour == 9 and \
                        cur_time_minute < 27 and \
                        num_orders_sent < BEFORE_OPEN_N_ORDERS:
                    return True
        return False

    def process_symbol_dict(self, symbol_dict: dict) -> dict:
        symbol = symbol_dict[SYMBOL]
        print(f'Start processing symbol: {symbol}')
        symbol_prop = self.get_tier_prop(symbol=symbol)
        # If the tier proportion is not 0 (stock is in black list)
        if float(symbol_prop):
            # Make trade decision
            try:
                # Get indicators
                indicators = self.__factors[symbol]
                main_etf = self.__main_etfs[symbol]
                factors_l1 = list(
                    map(lambda x: current_percentage(
                        self.__stocks_l1.get(x)), indicators))
                main_etf = [current_percentage(self.__stocks_l1.get(main_etf))]

                if INIT_PCT not in factors_l1:
                    valid_tier = self.validate_tier(symbol=symbol)

                    if valid_tier:
                        model_dict = self.__models[symbol]
                        model = model_dict[MODEL]
                        model_main_etf = model_dict[MODEL_MAIN_ETF]
                        pct_bid_net = symbol_dict[PCT_BID_NET]
                        pct_ask_net = symbol_dict[PCT_ASK_NET]
                        bid_l1 = symbol_dict[BID]
                        ask_l1 = symbol_dict[ASK]
                        bid_venue = symbol_dict[BID_VENUE]
                        # bid_venue = 1
                        ask_venue = symbol_dict[ASK_VENUE]
                        # ask_venue = 1
                        close = symbol_dict[CLOSE]
                        vwap = symbol_dict[VWAP]
                        prem_high = symbol_dict[PREM_HIGH]
                        prem_low = symbol_dict[PREM_LOW]
                        imb = symbol_dict[IMB]
                        vol = symbol_dict[VOL]
                        pred_array = np.array(factors_l1).reshape(1, -1)
                        main_etf_pred_array = np.array(main_etf).reshape(1, -1)
                        prediction = model.predict(pred_array)[0]
                        prediction_main_etf = model_main_etf.predict(main_etf_pred_array)[0]
                        std_err = model_dict[MAE]
                        std_err_main_etf = model_dict[MAE_MAIN_ETF]
                        lower_sigma = model_dict[LOWER_SIGMA]
                        upper_sigma = model_dict[UPPER_SIGMA]
                        lower_two_sigma = model_dict[LOWER_TWO_SIGMA]
                        upper_two_sigma = model_dict[UPPER_TWO_SIGMA]

                        # Check for trade opportunity
                        symbol_sector = self.__stock_to_sector[symbol]
                        symbol_policy = self.__get_policy(sector=symbol_sector)
                        delta_long_coef, delta_short_coef = self.__get_deltas(policy=symbol_policy)
                        acc_info = self.__get_acc_info()
                        bp = float(acc_info[BP_KEY])
                        bp_usage_pct = float(acc_info[BP_USAGE_PCT_KEY])
                        order_data = get_order(prediction=prediction,
                                               prediction_main_etf=prediction_main_etf,
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
                                               vwap=vwap,
                                               prem_high=prem_high,
                                               prem_low=prem_low,
                                               imb=imb,
                                               vol=vol,
                                               std_err=std_err,
                                               std_err_main_etf=std_err_main_etf,
                                               lower_sigma=lower_sigma,
                                               upper_sigma=upper_sigma,
                                               lower_two_sigma=lower_two_sigma,
                                               upper_two_sigma=upper_two_sigma,
                                               policy=symbol_policy,
                                               prop=symbol_prop,
                                               delta_long_coef=delta_long_coef,
                                               delta_short_coef=delta_short_coef,
                                               bp=bp,
                                               bp_usage_pct=bp_usage_pct)
                        if order_data:
                            if self.__sent_orders_by_ticker.get(symbol):
                                self.__sent_orders_by_ticker[symbol] += 1
                            else:
                                self.__sent_orders_by_ticker[symbol] = 1
                            self.__redis_connector.h_set_int(h=SENT_ORDERS_BY_TICKER,
                                                             key=symbol,
                                                             value=self.__sent_orders_by_ticker[symbol])
                        return order_data

                else:
                    raise TypeError('One of indicators is not populated yet')

            except KeyError as e:
                message = f'Process symbol dict error: ' \
                          f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                print(message)
                print(traceback.format_exc())
                print(f'Failed to make inference on symbol message: {symbol_dict}')
                pass

            except TypeError as e:
                message = f'Process symbol dict error: ' \
                          f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
                print(message)
                print(traceback.format_exc())
                print(f'Failed to make inference on symbol message: {symbol_dict}')
                pass

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
