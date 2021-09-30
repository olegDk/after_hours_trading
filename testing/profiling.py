import cProfile
import pstats
from pstats import SortKey
from analytics.trader import Trader
from datetime import datetime, timedelta
import json
import orjson
from line_profiler import LineProfiler

trader = Trader()

test_md_json = {"messageId": "marketData",
                "data": [
                    {"symbol": "SWKS", "close": 1001.85, "pctBidNet": -0.3430278491120655,
                     "pctAskNet": 0.7819032687719016, "bid": 1000.0, "ask": 1001.5, "bidVenue": 1, "askVenue": 1},
                    {"symbol": "COP", "close": 1009.51, "pctBidNet": -0.5733179612383288,
                     "pctAskNet": 0.15487884355065562, "bid": 1000.0, "ask": 1001.5, "bidVenue": 1, "askVenue": 1},
                    {"symbol": "OXY", "close": 995.85, "pctBidNet": -0.44386369725636365,
                     "pctAskNet": 0.8218819164979141, "bid": 1000.0, "ask": 1001.5, "bidVenue": 1, "askVenue": 1},
                    {"symbol": "BAC", "close": 991.09, "pctBidNet": -0.9347176586361323,
                     "pctAskNet": -0.49935774740509625, "bid": 1000.0, "ask": 1001.5, "bidVenue": 1,
                     "askVenue": 1}]}

market_data = test_md_json["data"]

test_news_json = {"messageId": "news", "symbol": "PDCE",
                  "content": "Q3 equalweight maintain guides Q3 Q4 equalweight "
                             "neutral Q3", "relevance": 9, "isYesterdaysAMC": 0,
                  "datetime": "09-25-21 20:48:21"}

test_order_report_json = {"cid": 1042,
                          "id": 3314487761576189735,
                          "messageId": "orderReport",
                          "price": "166.670",
                          "side": 84,
                          "size": 760,
                          "symbol": "SWKS"}


def test_process_md_message():
    lp = LineProfiler()
    lp_wrapper = lp(trader.process_md_message)
    lp_wrapper(test_md_json)
    lp.print_stats()


def test_process_symbol_dict():
    lp = LineProfiler()
    lp_wrapper = lp(trader.process_symbol_dict)
    lp_wrapper(test_md_json['data'][1])
    lp.print_stats()


def test_get_tier_prop():
    lp = LineProfiler()
    lp_wrapper = lp(trader.get_tier_prop)
    lp_wrapper('COP')
    lp.print_stats()


def test_validate_tier():
    lp = LineProfiler()
    lp_wrapper = lp(trader.validate_tier)
    lp_wrapper('BIDU')
    lp.print_stats()


def test_send_md_to_mq():
    lp = LineProfiler()
    lp_wrapper = lp(trader.send_market_data_to_mq)
    lp_wrapper(market_data)
    lp.print_stats()


def test_process_order_report():
    lp = LineProfiler()
    lp_wrapper = lp(trader.process_order_report)
    lp_wrapper(test_order_report_json)
    lp.print_stats()


# test_process_md_message()
# test_process_symbol_dict()
# test_get_tier_prop()
# test_validate_tier()
# test_send_md_to_mq()
test_process_order_report()
