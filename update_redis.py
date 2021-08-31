import os
import pickle
import json
from messaging.redis_connector import RedisConnector
from config.constants import *

r = RedisConnector()


def get_sectors() -> list:
    sectors_path = 'analytics/modeling/sectors'
    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]
    sectors = list(map(lambda x: x.split('/')[-1], sectors_dirs))

    return sectors


def get_tickers() -> list:
    sectors_path = 'analytics/modeling/sectors'
    sectors = []
    sectors_dirs = \
        [f.path for f in os.scandir(sectors_path) if f.is_dir()]
    all_tickers = []

    for sector_dir in sectors_dirs:
        sector = sector_dir.split('/')[-1]
        if 'tickers' not in os.listdir(sector_dir):
            print(f'tickers dir missing '
                  f'for sector: {sector}')
            continue

        try:
            with open(f'{sector_dir}/tickers/'
                      f'traidable_tickers_{sector}.pkl',
                      'rb') as inp:
                tickers = pickle.load(inp)
                all_tickers = all_tickers + tickers

        except Exception as e:
            print(e)
            print(f'Failed to load tickers for sector: {sector}')
            continue

    return all_tickers


def set_sector_policy(sector: str,
                      policy: str):
    sectors = get_sectors()
    valid_policies = [NEUTRAL, AGG_BULL, AGG_BEAR, BULL, BEAR]
    if sector not in sectors:
        print(f'Invalid sector: {sector}, should be one of: '
              f'{sectors}')
    if policy not in valid_policies:
        print(f'Invalid policy: {policy}, should be one of: '
              f'{valid_policies}')
    r.h_set_str(h=POLICY,
                key=sector,
                value=policy)


def set_stock_prop(stock: str,
                   prop: float):
    stocks = get_tickers()
    if stock not in stocks:
        print(f'Invalid stock {stock}, should be one of: '
              f'{stocks}')
    r.h_set_float(h=STOCK_TO_TIER_PROPORTION,
                  key=stock,
                  value=prop)


def get_sector_policy(sector: str) -> str:
    sectors = get_sectors()
    sector_policy = ''
    if sector not in sectors:
        print(f'Invalid sector: {sector}, should be one of: '
              f'{sectors}')
    sector_policy = r.hm_get(h=POLICY,
                             key=sector)[0]

    return sector_policy


def get_stock_prop(stock: str) -> float:
    stocks = get_tickers()
    stock_prop = 1.0
    if stock not in stocks:
        print(f'Invalid stock: {stock}, should be one of: '
              f'{stocks}')
        # stock_prop = float(r.hm_get(h=STOCK_TO_TIER_PROPORTION,
        #                             key=stock)[0])
    stock_prop = r.hm_get(h=STOCK_TO_TIER_PROPORTION,
                          key=stock)[0]

    return stock_prop


def get_acc_info() -> dict:
    acc_info = r.h_getall(h=ACCOUNT_INFORMATION)
    return acc_info


def set_acc_info(bp: float,
                 bp_usage: float):
    if bp_usage < 0 or bp_usage > 1 or bp < 0:
        print(f'Invalid arguments: bp: {bp}, '
              f'bp_usage: {bp_usage}')
    acc_policy = {
        BP_KEY: bp,
        BP_USAGE_PCT_KEY: bp_usage
    }
    r.set_dict(name=ACCOUNT_INFORMATION,
               d=acc_policy)


#print(get_sector_policy(sector=APPLICATION_SOFTWARE))
#print(get_stock_prop(stock='AMZN'))
#print(get_acc_info())

#set_sector_policy(sector=APPLICATION_SOFTWARE,
#                  policy=NEUTRAL)
set_stock_prop(stock='FIVN',
               prop=0)
set_stock_prop(stock='BILI',
               prop=0)
set_stock_prop(stock='DOCU',
               prop=0)
#set_acc_info(bp=1000000.0,
#             bp_usage=0.5)

print(get_sector_policy(sector=APPLICATION_SOFTWARE))
print(type(get_stock_prop(stock='FIVN')))
print(get_acc_info())
