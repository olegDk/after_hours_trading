import numpy as np
import uuid
import config.messages as messages
from config.constants import *
import random
from analytics.base_trader import BaseTrader


def generate_id() -> str:
    unique_id = uuid.uuid4().__str__().strip()
    unique_id = unique_id.replace('-', '')
    return unique_id


def generate_cid() -> int:
    return random.getrandbits(64)


class AppTrader(BaseTrader):
    def __init__(self):
        super().__init__()
        print('Initializing AppTrader')
        self.__sent_orders_by_ticker = {}

    def get_order(self,
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
                  ask_venue,
                  std_err) -> dict:
        print(f'From AppTrader get order if exists for '
              f'symbol: {symbol}')
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
        trade_flag = delta_long >= std_err or delta_short <= -std_err  # change
        if trade_flag:
            side = BUY if np.sign(delta_long) > 0 else SELL
            order_params = side_params[side]
            order_related_data_dict = dict(zip(indicators, factors_l1))
            order_data[ORDER_RELATED_DATA] = order_related_data_dict
            target = float(close + float(prediction / 100) * close)
            order = messages.order_request()
            order[ORDER][DATA][SYMBOL] = symbol
            order[ORDER][DATA][PRICE] = order_params[PRICE]
            order[ORDER][DATA][SIDE] = side
            order[ORDER][DATA][SIZE] = 100
            order[ORDER][DATA][VENUE] = order_params[VENUE]
            order[ORDER][DATA][TARGET] = target  # change to target
            order[ORDER][CID] = generate_cid()
            order_data[ORDER_DATA] = order
            print(f'Stock: {symbol}, {side} '
                  f'{order_params[PRICE]},\n'
                  f'Current entry: {order_params[PCT_NET]}, '
                  f'prediction: {prediction}, '
                  f'target: {target}')

        # Change in future, logic to avoid order spamming
        num_orders_sent = self.__sent_orders_by_ticker.get(symbol)
        if num_orders_sent:
            if num_orders_sent >= 3:
                return {}
            else:
                self.__sent_orders_by_ticker[symbol] += 1
                return order_data
        else:
            self.__sent_orders_by_ticker[symbol] = 1
            return order_data
