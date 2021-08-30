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


def get_position_size(price: float,
                      bp: float,
                      prop: float) -> int:
    return int((bp*0.1*0.125*prop)/(price+1e-7))


class SemiTrader(BaseTrader):
    def __init__(self):
        super().__init__()
        print('Initializing SemiTrader')
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
                  std_err,
                  policy,
                  prop,
                  delta_long_coef,
                  delta_short_coef,
                  bp) -> dict:
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
        trade_flag = delta_long >= std_err * delta_long_coef or \
                     delta_short <= -std_err * delta_short_coef  # change
        if trade_flag:
            side = BUY if np.sign(delta_long) > 0 else SELL
            order_params = side_params[side]
            order_related_data_dict = dict(zip(indicators, factors_l1))
            order_related_data_dict.update({POLICY: policy,
                                            LONG_COEF: delta_long_coef,
                                            SHORT_COEF: delta_short_coef,
                                            'prediction': prediction})
            order_data[ORDER_RELATED_DATA] = order_related_data_dict
            target = float(close + float(prediction / 100) * close)
            order = messages.order_request()
            order[ORDER][DATA][SYMBOL] = symbol
            price = order_params[PRICE]
            order[ORDER][DATA][PRICE] = price
            order[ORDER][DATA][SIDE] = side
            order[ORDER][DATA][SIZE] = get_position_size(price=order_params[PRICE],
                                                         bp=bp,
                                                         prop=prop)
            order[ORDER][DATA][VENUE] = order_params[VENUE]
            order[ORDER][DATA][TARGET] = target  # change to target
            order[ORDER][CID] = generate_cid()
            order_data[ORDER_DATA] = order
            print(f'Stock: {symbol}, {side} '
                  f'{order_params[PRICE]},\n'
                  f'Current entry: {order_params[PCT_NET]}, '
                  f'prediction: {prediction}, '
                  f'target: {target}')

        return order_data
