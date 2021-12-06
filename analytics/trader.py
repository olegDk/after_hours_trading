import numpy as np
import sys
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
import operator

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
cwd = os.getcwd()


def generate_id() -> str:
    unique_id = uuid.uuid4().__str__().strip()
    unique_id = unique_id.replace('-', '')
    return unique_id


def generate_cid() -> int:
    return random.getrandbits(64)


def get_position_size(price: float,
                      bp: float,
                      bp_usage_pct: float,
                      prop: float,
                      sector_prop: float,
                      order_side: str,
                      position: dict
                      ) -> int:
    size = int((bp * bp_usage_pct * 0.1 * 0.125 * prop * sector_prop) / (price + 1e-7))
    if size < 5:
        size = 5
    elif 95 <= size < 100:
        size = 100
    if position:
        pos_side = position[SIDE]
        pos_size = position[SIZE]
        opposite_side = not pos_side == order_side
        open_position = not pos_size == 0
        # if flipping
        if opposite_side and open_position:
            return 0

    return size


def current_percentage(l1_dict: dict,
                       return_missing: bool = False) -> float:
    if (not l1_dict) and return_missing:
        return INIT_PCT
    elif not l1_dict:
        raise KeyError('Empty l1 data')
    try:
        pct_bid_net = l1_dict[PCT_BID_NET]
        pct_ask_net = l1_dict[PCT_ASK_NET]
        if pct_bid_net >= 0:
            return pct_bid_net
        elif pct_ask_net <= 0:
            return pct_ask_net
        else:
            return 0
    except KeyError:
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


def get_reports() -> list:
    # Getting earnings calendar manually every day
    # https://hosting.briefing.com/cschwab/
    # Calendars/EarningsCalendar5Weeks.htm
    reports_path = 'analytics/modeling/reports'
    previous_day_reports_path = 'analytics/modeling/previous_day_reports'
    if sys.gettrace():
        reports_path = '/home/oleh/takion_trader/analytics/modeling/reports'
        previous_day_reports_path = 'analytics/modeling/previous_day_reports'

    with open(reports_path) as i:
        reports = i.read().splitlines()
        print(f'Reports: {reports}')

    with open(previous_day_reports_path) as i:
        previous_day_reports = i.read().splitlines()
        reports = reports + previous_day_reports

    return reports


def get_sectors_reports(stock_to_sector: dict,
                        reports: list) -> dict:
    sectors_reports = {}

    for ticker in reports:
        sector_report = stock_to_sector.get(ticker)
        if sector_report:
            if sector_report in sectors_reports:
                sectors_reports[sector_report] = sectors_reports[sector_report] + [ticker]
            else:
                sectors_reports[sector_report] = [ticker]

    return sectors_reports


def get_sector_stocks_maps() -> Tuple[dict, dict]:
    sectors_path = 'analytics/modeling/sectors'
    if sys.gettrace():
        sectors_path = '/home/oleh/takion_trader/analytics/modeling/sectors'

    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]

    sector_to_stocks = {}
    stock_to_sector = {}

    for sector_dir in sectors_dirs:
        sector = sector_dir.split('/')[-1]

        try:
            with open(f'{sector_dir}/tickers/traidable_tickers_{sector}.pkl', 'rb') as inp:
                stocks = pickle.load(inp)

            sector_to_stocks.update({sector: stocks})
            stock_to_sector.update({stock: sector for stock in stocks})

        except Exception as e:
            message = f'Get sector stocks maps error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to get maps for sector: {sector}')
            continue

    return sector_to_stocks, stock_to_sector


def get_main_stock(sector_reports: list,
                   market_cap_data: dict) -> str:
    market_caps = {sector_report: market_cap_data.get(sector_report)
                   for sector_report in sector_reports
                   if market_cap_data.get(sector_report) is not None}

    if market_caps:
        return max(market_caps.items(), key=operator.itemgetter(1))[0]

    return ''


def get_market_cap_data() -> dict:
    market_cap_data_path = f'analytics/modeling/market_cap_data.pkl'
    if sys.gettrace():
        market_cap_data_path = f'/home/oleh/takion_trader/analytics/' \
                               f'modeling/market_cap_data.pkl'
    with open(market_cap_data_path, 'rb') as i:
        market_cap_data = pickle.load(i)
    return market_cap_data


def init_models(reports: list,
                market_cap_data: dict) -> Tuple[dict, dict, dict, dict, dict]:
    print('Initializing models...')

    sectors_path = 'analytics/modeling/sectors'
    if sys.gettrace():
        sectors_path = '/home/oleh/takion_trader/analytics/modeling/sectors'

    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]

    sector_to_stocks, stock_to_sector = get_sector_stocks_maps()
    sectors_reports = \
        get_sectors_reports(stock_to_sector=stock_to_sector,
                            reports=reports)

    factors_dict = {}
    main_etfs_dict = {}
    sector_to_indicators = {}
    models_dict = {}

    for sector_dir in sectors_dirs:
        sector = sector_dir.split('/')[-1]
        print(f'Sector: {sector}...')

        if sector == 'Software':
            print()

        sector_reports = sectors_reports.get(sector)
        if sector_reports:
            main_stock = get_main_stock(sector_reports,
                                        market_cap_data)
            if main_stock:
                tickers_indicators_path \
                    = f'{sector_dir}/models/report_models/' \
                      f'{main_stock}/tickers_indicators.pkl'
                main_etfs_path = f'{sector_dir}/models/report_models/' \
                                 f'{main_stock}/tickers_main_etf.pkl'
                models_path = f'{sector_dir}/models/report_models/' \
                              f'{main_stock}/tickers_models.pkl'
        else:
            tickers_indicators_path \
                = f'{sector_dir}/models/tickers_indicators.pkl'
            main_etfs_path = f'{sector_dir}/models/tickers_main_etf.pkl'
            models_path = f'{sector_dir}/models/tickers_models.pkl'

        try:
            with open(tickers_indicators_path, 'rb') as i:
                tickers_indicators = pickle.load(i)
            factors_dict.update(tickers_indicators)
            with open(main_etfs_path, 'rb') as i:
                main_etfs = pickle.load(i)
            main_etfs_dict.update(main_etfs)
            with open(models_path, 'rb') as i:
                models = pickle.load(i)
            models_dict.update(models)
            key = list(tickers_indicators.keys())[0]
            current_sector_indicators = tickers_indicators[key]
            sector_to_indicators.update({sector: current_sector_indicators})

        except Exception as e:
            message = f'Init factors error: ' \
                      f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
            print(message)
            print(traceback.format_exc())
            print(f'Failed to init factors for sector {sector}')
            continue

    # Filter stocks with reports
    factors_dict_filtered = {key: factors_dict[key] for key in factors_dict
                             if key not in reports}
    main_etfs_dict_filtered = {key: main_etfs_dict[key] for key in main_etfs_dict
                               if key not in reports}
    stocks_to_sector_filtered = {key: stock_to_sector[key] for key in stock_to_sector
                                 if key not in reports}
    models_dict_filtered = {key: models_dict[key] for key in models_dict
                            if key not in reports}

    return factors_dict_filtered, main_etfs_dict_filtered, stocks_to_sector_filtered, \
           sector_to_indicators, models_dict_filtered


def adjust_limit_price(side,
                       l1_price,
                       target,
                       prem_low,
                       prem_high,
                       vwap,
                       std_err_main_etf,
                       premarket_delta,
                       close) -> float:
    if vwap:  # and (prem_low <= vwap <= prem_high):  # if there were another trades

        # short logic
        if side == SELL:
            if premarket_delta < 0:
                l1_price = l1_price - premarket_delta
            short_bound = target + close * 2 * std_err_main_etf / 3 / 100
            if prem_low <= l1_price <= prem_high:
                if l1_price >= vwap:
                    return max([l1_price, short_bound])
                else:
                    adjusted_price = round(vwap - float(np.abs(vwap - target)) / 3, 2)
                    return max([adjusted_price, l1_price, short_bound])
            elif l1_price >= prem_high:
                return max([l1_price, short_bound])
            elif l1_price <= prem_low:
                adjusted_price = round(vwap - float(np.abs(vwap - target)) / 3, 2)
                return max([adjusted_price, l1_price, short_bound])

        # long logic
        elif side == BUY:
            if premarket_delta > 0:
                l1_price = l1_price - premarket_delta
            long_bound = target - close * 2 * std_err_main_etf / 3 / 100
            if prem_low <= l1_price <= prem_high:
                if l1_price <= vwap:
                    return min([l1_price, long_bound])
                else:
                    adjusted_price = round(vwap + float(np.abs(vwap - target)) / 3, 2)
                    return min([adjusted_price, l1_price, long_bound])
            elif l1_price <= prem_low:
                return min([l1_price, long_bound])
            elif l1_price >= prem_high:
                adjusted_price = round(vwap + float(np.abs(vwap - target)) / 3, 2)
                return min([adjusted_price, l1_price, long_bound])

    return l1_price


def get_nearest_significant_delta(stock_snapshots: dict,
                                  current_stock_percentage: float) -> float:
    significant_delta = 0.0001
    now_dt = datetime.now(tz=EST)
    # For testing
    # now_dt = datetime(year=2021,
    #                   month=12,
    #                   day=6,
    #                   hour=9,
    #                   minute=20,
    #                   second=33)
    minutes_ago = list(range(1, 31))
    dts = [now_dt - timedelta(minutes=minutes) for minutes in minutes_ago]
    dts_keys = [f'{dt.hour}_{dt.minute}' for dt in dts]
    stock_snapshots_deltas = [current_stock_percentage - current_percentage(stock_snapshots.get(dt_key),
                                                                            return_missing=True)
                              for dt_key in dts_keys]
    nearest_delta_list = [v for v in stock_snapshots_deltas if significant_delta < np.abs(v) < 900]
    if nearest_delta_list:
        return nearest_delta_list[0]

    return 0.0


class Trader:
    def __init__(self):
        self.__rabbit_sender = RabbitSender(RABBIT_MQ_HOST, RABBIT_MQ_PORT)
        self.__redis_connector = RedisConnector(REDIS_HOST, REDIS_PORT)
        self.__reports = \
            get_reports()
        self.__factors, \
        self.__main_etfs, \
        self.__stock_to_sector, \
        self.__sector_to_indicators, \
        self.__models \
            = init_models(reports=self.__reports,
                          market_cap_data=get_market_cap_data())
        self.__subscription_list, self.__all_indicators = self.get_subscription_list()
        self.__stocks_l1 = self.init_stocks_data()
        self.__init_policy()
        self.positions = {}
        self.__orders = {}
        self.__sent_orders_by_ticker = {}
        self.__na = NewsAnalyzer()

    def init_stocks_data(self) -> dict:
        print('Inializing indicators...')
        now_dt = datetime.now(tz=EST)
        # Update to get from subscription response or request directly
        indicators_dict = {indicator: {L1_DATA: {PCT_BID_NET: INIT_PCT,
                                                 PCT_ASK_NET: INIT_PCT},
                                       STOCK_SNAPSHOT: {
                                           f'{now_dt.hour}_{now_dt.minute}': {
                                               PCT_BID_NET: INIT_PCT,
                                               PCT_ASK_NET: INIT_PCT
                                           }
                                       }}
                           for indicator in self.__all_indicators}

        print(indicators_dict)

        return indicators_dict

    def get_subscription_list(self) -> Tuple[list, list]:

        all_tickers_to_subscribe = []
        all_indicators = []
        for key in self.__factors.keys():
            tickers_to_subscribe = list(set([key] + self.__factors[key]))
            all_indicators = all_indicators + self.__factors[key]
            all_tickers_to_subscribe = all_tickers_to_subscribe + tickers_to_subscribe

        return list(set(all_tickers_to_subscribe)), list(set(all_indicators))

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
            now_dt = datetime.now(tz=EST)
            pct_bid_net = symbol_dict[PCT_BID_NET]
            pct_ask_net = symbol_dict[PCT_ASK_NET]
            l1_dict = self.__stocks_l1.get(symbol)
            if not l1_dict:
                stock_dict = {
                    L1_DATA: {PCT_BID_NET: pct_bid_net,
                              PCT_ASK_NET: pct_ask_net},
                    STOCK_SNAPSHOT: {
                        f'{now_dt.hour}_{now_dt.minute}': {
                            PCT_BID_NET: pct_bid_net,
                            PCT_ASK_NET: pct_ask_net
                        }
                    }
                }
                self.__stocks_l1[SYMBOL] = stock_dict

            else:
                self.__stocks_l1[symbol][L1_DATA][PCT_BID_NET] = pct_bid_net
                self.__stocks_l1[symbol][L1_DATA][PCT_ASK_NET] = pct_ask_net
                self.__stocks_l1[symbol][STOCK_SNAPSHOT][f'{now_dt.hour}_{now_dt.minute}'] = {
                    PCT_BID_NET: pct_bid_net,
                    PCT_ASK_NET: pct_ask_net
                }
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
                    if pos_size == 0:
                        self.__sent_orders_by_ticker[symbol] = 0
                # if flip
                elif size > pos_size:
                    pos_size = int(np.abs(pos_size - size))
                    investment = pos_size * price
                    self.positions[symbol][SIZE] = pos_size
                    self.positions[symbol][SIDE] = side
                    self.positions[symbol][PRICE] = price
                    self.positions[symbol][INVESTMENT] = investment
                    # Recalculating number of sent tiers
                    bp_info = self.__get_acc_info()
                    bp = float(bp_info[BP_KEY])
                    bp_usage_pct = float(bp_info[BP_USAGE_PCT_KEY])
                    cur_bp_per_tier = bp * bp_usage_pct * 0.1 / MAX_ORDERS
                    n_tiers_symbol = int(investment / cur_bp_per_tier)
                    if symbol in self.__sent_orders_by_ticker:
                        self.__sent_orders_by_ticker[symbol] = \
                            self.__sent_orders_by_ticker[symbol] + n_tiers_symbol
                    else:
                        self.__sent_orders_by_ticker[symbol] = n_tiers_symbol

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

        sectors = list(self.__sector_to_indicators.keys())

        policy_dict = {sector: NEUTRAL for sector in sectors}

        delta_dict = {NEUTRAL: {LONG_COEF: 1,
                                SHORT_COEF: 1},
                      BULL: {LONG_COEF: 1,
                             SHORT_COEF: 2},
                      BEAR: {LONG_COEF: 2,
                             SHORT_COEF: 1},
                      AGG_BULL: {LONG_COEF: 1 / 2,
                                 SHORT_COEF: 2},
                      AGG_BEAR: {LONG_COEF: 2,
                                 SHORT_COEF: 1 / 2},
                      DELTA_FREE_BULL: {LONG_COEF: 0,
                                        SHORT_COEF: 4},
                      DELTA_FREE_BEAR: {LONG_COEF: 4,
                                        SHORT_COEF: 0}
                      }
        sector_proportion_dict = {sector: 0 for sector in sectors}
        sector_side_policy = {sector: BOTH for sector in sectors}
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
        self.__redis_connector.set_dict(name=SECTOR_PROPORTION,
                                        d=sector_proportion_dict)
        self.__redis_connector.set_dict(name=SECTOR_SIDE_POLICY,
                                        d=sector_side_policy)
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

    def __get_position(self, ticker: str) -> dict:
        position_str = self.__redis_connector.hm_get(h=POSITIONS, key=ticker)[0]
        if not position_str:
            return {}
        return json.loads(position_str)

    def get_tier_prop(self, symbol: str) -> float:
        prop = self.__redis_connector.hm_get(h=STOCK_TO_TIER_PROPORTION,
                                             key=symbol)[0]
        if prop:
            prop = float(prop)
        else:
            prop = 0.0
        return prop

    def get_sector_prop(self, sector: str) -> float:
        prop = self.__redis_connector.hm_get(h=SECTOR_PROPORTION,
                                             key=sector)[0]
        if prop:
            prop = float(prop)
        else:
            prop = 0.0
        return prop

    def get_sector_side_policy(self, sector: str) -> str:
        policy = self.__redis_connector.hm_get(h=SECTOR_SIDE_POLICY, key=sector)[0]
        if not policy:
            return BOTH
        return policy

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
        cur_time = datetime.now(EST)
        cur_time_hour = cur_time.hour
        cur_time_minute = cur_time.minute
        all_invalid_flag = cur_time > final_dt
        if not all_invalid_flag:
            if not num_orders_sent:
                return True
            if num_orders_sent < MAX_ORDERS:
                if cur_time_hour < 9 and \
                        num_orders_sent < BEFORE_9_N_ORDERS:
                    return True
                elif cur_time_hour == 9 and \
                        cur_time_minute < 15 and \
                        num_orders_sent < BEFORE_9_15_N_ORDERS:
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
        symbol_sector = self.__stock_to_sector[symbol]
        sector_prop = self.get_sector_prop(sector=symbol_sector)
        # If the tier proportion is not 0 (stock is in black list)
        if float(symbol_prop) and float(sector_prop):
            # Make trade decision
            try:
                # Get indicators
                indicators = self.__factors[symbol]
                main_etf = self.__main_etfs[symbol]
                factors_l1 = list(
                    map(lambda x: current_percentage(
                        self.__stocks_l1.get(x).get(L1_DATA)), indicators))
                main_etf = [current_percentage(self.__stocks_l1.get(main_etf).get(L1_DATA))]

                if INIT_PCT not in factors_l1:
                    # valid_tier = self.validate_tier(symbol=symbol)
                    valid_tier = True if random.random() > 0.9 else False
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
                        sector_side = self.get_sector_side_policy(sector=symbol_sector)
                        symbol_policy = self.__get_policy(sector=symbol_sector)
                        delta_long_coef, delta_short_coef = self.__get_deltas(policy=symbol_policy)
                        acc_info = self.__get_acc_info()
                        bp = float(acc_info[BP_KEY])
                        bp_usage_pct = float(acc_info[BP_USAGE_PCT_KEY])
                        if bp_usage_pct:
                            order_data = self.get_order(prediction=prediction, prediction_main_etf=prediction_main_etf,
                                                        pct_bid_net=pct_bid_net, pct_ask_net=pct_ask_net,
                                                        indicators=indicators, factors_l1=factors_l1,
                                                        main_etf_l1=main_etf,
                                                        close=close, symbol=symbol, bid_l1=bid_l1, ask_l1=ask_l1,
                                                        bid_venue=bid_venue, ask_venue=ask_venue, vwap=vwap,
                                                        prem_high=prem_high, prem_low=prem_low, imb=imb, vol=vol,
                                                        std_err=std_err, std_err_main_etf=std_err_main_etf,
                                                        lower_sigma=lower_sigma, upper_sigma=upper_sigma,
                                                        lower_two_sigma=lower_two_sigma,
                                                        upper_two_sigma=upper_two_sigma,
                                                        policy=symbol_policy, prop=symbol_prop,
                                                        delta_long_coef=delta_long_coef,
                                                        delta_short_coef=delta_short_coef,
                                                        bp=bp, bp_usage_pct=bp_usage_pct,
                                                        sector_prop=sector_prop, sector_side=sector_side)
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

    def get_premarket_delta(self,
                            symbol: str) -> float:
        symbol_l1_data = self.__stocks_l1.get(symbol)
        if symbol_l1_data:
            stock_snapshots = symbol_l1_data.get(STOCK_SNAPSHOT)
            current_stock_percentage = current_percentage(symbol_l1_data.get(L1_DATA))
            delta = get_nearest_significant_delta(stock_snapshots=stock_snapshots,
                                                  current_stock_percentage=current_stock_percentage)
            return delta
        else:
            return 0.0

    def get_order(self,
                  prediction,
                  prediction_main_etf,
                  pct_bid_net,
                  pct_ask_net,
                  indicators,
                  factors_l1,
                  main_etf_l1,
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
                  bp_usage_pct,
                  sector_prop,
                  sector_side) -> dict:
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
        trade_flag = True
        if trade_flag:
            print('Trade flag!!!=================================================================================')
            side = BUY if np.sign(delta_long) > 0 else SELL
            side_match = (side == BUY and sector_side == LONG_ONLY) or \
                         (side == SELL and sector_side == SHORT_ONLY) or \
                         sector_side == BOTH
            print(f'Side match: {side_match}')
            if side_match:
                print('Side flag!!!=================================================================================')
                position = self.__get_position(ticker=symbol)
                order_params = side_params[side]
                order_related_data_dict = dict(zip(indicators, factors_l1))
                order_related_data_dict.update({POLICY: policy,
                                                LONG_COEF: delta_long_coef,
                                                SHORT_COEF: delta_short_coef,
                                                PREDICTION_KEY: prediction})
                order_data[ORDER_RELATED_DATA] = order_related_data_dict
                target_pct = (prediction_main_etf + main_etf_l1) / 2
                target = float(close + float(target_pct / 100) * close)
                order = messages.order_request()
                order[ORDER][DATA][SYMBOL] = symbol
                price = order_params[PRICE]
                # Get premarket deltas
                premarket_delta = self.get_premarket_delta(symbol=symbol)
                order_related_data_dict.update({PREMARKET_DELTA: premarket_delta,
                                                SECTOR_SIDE_POLICY: sector_side,
                                                SECTOR_PROPORTION: sector_prop})
                order[ORDER][DATA][LIMIT] = adjust_limit_price(side=side,
                                                               l1_price=price,
                                                               target=target,
                                                               prem_low=prem_low,
                                                               prem_high=prem_high,
                                                               vwap=vwap,
                                                               std_err_main_etf=std_err_main_etf,
                                                               premarket_delta=premarket_delta,
                                                               close=close)
                order[ORDER][DATA][SIDE] = side
                position_size = get_position_size(price=order_params[PRICE],
                                                  bp=bp,
                                                  bp_usage_pct=bp_usage_pct,
                                                  prop=prop,
                                                  sector_prop=sector_prop,
                                                  order_side=side,
                                                  position=position)
                if not position_size:
                    return {}
                order[ORDER][DATA][SIZE] = position_size
                order[ORDER][DATA][VENUE] = order_params[VENUE]
                order[ORDER][DATA][TARGET] = target
                order[ORDER][CID] = generate_cid()
                order_data[ORDER_DATA] = order
                print(order_data)
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        return order_data


stock_snapshot = {
    "8_42": {
        "pctBidNet": -3.162812584027973,
        "pctAskNet": -2.996644295302016
    },
    "8_43": {
        "pctBidNet": -2.8999597693442523,
        "pctAskNet": -2.7552728490123934
    },
    "8_44": {
        "pctBidNet": -3.169747899159666,
        "pctAskNet": -3.048514352862173
    },
    "8_45": {
        "pctBidNet": -3.6050632911392424,
        "pctAskNet": -3.3469140375096798
    },
    "8_46": {
        "pctBidNet": -3.5351661325687367,
        "pctAskNet": -3.3434343434343456
    },
    "8_47": {
        "pctBidNet": -3.3434343434343456,
        "pctAskNet": -3.169747899159666
    },
    "8_48": {
        "pctBidNet": -3.503743171241649,
        "pctAskNet": -3.2738896366083505
    },
    "8_49": {
        "pctBidNet": -3.3503939659236273,
        "pctAskNet": -3.2565180824222058
    },
    "8_50": {
        "pctBidNet": -3.2738896366083505,
        "pctAskNet": -3.1974984869880942
    },
    "8_51": {
        "pctBidNet": -3.169747899159666,
        "pctAskNet": -3.0381361622129757
    },
    "8_52": {
        "pctBidNet": -3.1177557534016564,
        "pctAskNet": -3.0450547236956935
    },
    "8_53": {
        "pctBidNet": -3.169747899159666,
        "pctAskNet": -3.0831234256926976
    },
    "8_54": {
        "pctBidNet": -3.20443846671151,
        "pctAskNet": -3.0831234256926976
    },
    "8_55": {
        "pctBidNet": -3.0831234256926976,
        "pctAskNet": -3.0001006745192784
    },
    "8_56": {
        "pctBidNet": -2.996644295302016,
        "pctAskNet": -2.927565392354131
    },
    "8_57": {
        "pctBidNet": -2.996644295302016,
        "pctAskNet": -2.8241206030150776
    },
    "8_58": {
        "pctBidNet": -2.8241206030150776,
        "pctAskNet": -2.7655941339940373
    },
    "8_59": {
        "pctBidNet": -2.841346959289669,
        "pctAskNet": -2.7552728490123934
    },
    "9_0": {
        "pctBidNet": -3.065815983881798,
        "pctAskNet": -3.017386050882731
    },
    "9_1": {
        "pctBidNet": -3.2391523713420773,
        "pctAskNet": -3.1732159064170213
    },
    "9_2": {
        "pctBidNet": -3.3434343434343456,
        "pctAskNet": -3.2738896366083505
    },
    "9_3": {
        "pctBidNet": -3.2217925004203742,
        "pctAskNet": -3.0831234256926976
    },
    "9_4": {
        "pctBidNet": -3.2183212267958003,
        "pctAskNet": -3.0831234256926976
    },
    "9_5": {
        "pctBidNet": -3.065815983881798,
        "pctAskNet": -3.017386050882731
    },
    "9_6": {
        "pctBidNet": -3.1905594405594435,
        "pctAskNet": -3.017386050882731
    },
    "9_7": {
        "pctBidNet": -3.169747899159666,
        "pctAskNet": -3.100436681222714
    },
    "9_8": {
        "pctBidNet": -3.169747899159666,
        "pctAskNet": -3.065815983881798
    },
    "9_9": {
        "pctBidNet": -2.9620932572962038,
        "pctAskNet": -2.8241206030150776
    },
    "9_10": {
        "pctBidNet": -2.6521739130434803,
        "pctAskNet": -2.4944900821478755
    },
    "9_11": {
        "pctBidNet": -2.5698436037962833,
        "pctAskNet": -2.378252168112073
    },
    "9_12": {
        "pctBidNet": -2.241838774150572,
        "pctAskNet": -2.183973099843527
    },
    "9_13": {
        "pctBidNet": -2.204388798241818,
        "pctAskNet": -2.1227749126601214
    },
    "9_14": {
        "pctBidNet": -2.432919503404092,
        "pctAskNet": -2.3475274267231385
    },
    "9_15": {
        "pctBidNet": -2.6109922439154865,
        "pctAskNet": -2.4808013355592675
    },
    "9_16": {
        "pctBidNet": -2.4705371749073666,
        "pctAskNet": -2.378252168112073
    },
    "9_17": {
        "pctBidNet": -2.4671162449088553,
        "pctAskNet": -2.395329441201003
    },
    "9_18": {
        "pctBidNet": -2.422664931424564,
        "pctAskNet": -2.241838774150572
    },
    "9_19": {
        "pctBidNet": -2.3100000000000023,
        "pctAskNet": -2.2248126561199024
    },
    "9_20": {
        "pctBidNet": -2.3100000000000023,
        "pctAskNet": -2.2316224228091754
    },
    "9_21": {
        "pctBidNet": -2.460275070102819,
        "pctAskNet": -2.3100000000000023
    },
    "9_22": {
        "pctBidNet": -2.235027646392638,
        "pctAskNet": -2.160165091199577
    },
    "9_23": {
        "pctBidNet": -2.3065897803406608,
        "pctAskNet": -2.160165091199577
    },
    "9_24": {
        "pctBidNet": -2.269092363054779,
        "pctAskNet": -2.207792207792206
    },
    "9_25": {
        "pctBidNet": -2.772476142641899,
        "pctAskNet": -2.1941799294133424
    }
}


delta = get_nearest_significant_delta(stock_snapshots=stock_snapshot,
                                      current_stock_percentage=-2.23)
print()

# trader = Trader()
# print(trader.get_subscription_list())
# import pandas as pdT
# pd.DataFrame(trader.get_subscription_list()).to_csv('analytics/modeling/tickers.csv')
