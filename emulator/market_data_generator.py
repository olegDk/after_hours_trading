import pickle
import os
import messages
from random import uniform, randint
import numpy as np


def get_tickers() -> dict:
    path = 'analytics/modeling'
    list_subfolders_with_paths =\
        [f.path for f in os.scandir(f'{path}/sectors') if f.is_dir()]

    tickers = []

    with open(f'{path}/all_indicators.pkl', "rb") as input:
        f_list = pickle.load(input)
        tickers = tickers + f_list

    for sector_dir in list_subfolders_with_paths:
        for f in os.listdir(f'{sector_dir}/tickers'):
            with open(f'{sector_dir}/tickers/{f}', "rb") as input:
                if f.startswith('traidable_tickers'):
                    f_list = pickle.load(input)
                    tickers = tickers + f_list

    # Removing duplicates
    tickers = list(set(tickers))

    # Manually approximate sum of probabilities for most liquid stocks
    sum_liquid = 0.45

    # Determine probability mass to distribute across other tickers
    sum_to_distribute = float(1 - sum_liquid)

    # Uniformly distribute probability mass across rest of the tickers
    uniform_prob = float(sum_to_distribute/(len(tickers) - 4))

    uniform_dist = [uniform_prob for _ in tickers]

    prob_dict = dict(zip(tickers, uniform_dist))

    # Manually assigning probabilities for most liquid stocks
    prob_dict['SPY'] = 0.2
    prob_dict['QQQ'] = 0.1
    prob_dict['BAC'] = 0.1
    prob_dict['DIA'] = 0.05

    return prob_dict


class MarketDataGenerator:
    def __init__(self):
        self.__tickers_dict = get_tickers()

    def sample_l1_update(self) -> dict:
        sample_dict = messages.market_data()

        # Sample ticker, pctBidNet and pctAskNet
        tickers = list(self.__tickers_dict.keys())
        probs = list(self.__tickers_dict.values())

        n_tickers = randint(1, 6)
        sampled_tickers = np.random.choice(tickers,
                                           size=n_tickers,
                                           replace=False,
                                           p=probs)

        sampled_l1_dicts = list(map(self.__sample_stock_l1, sampled_tickers))
        sample_dict['data'] = sampled_l1_dicts

        return sample_dict

    def __sample_stock_l1(self,
                          ticker: str) -> dict:

        sample_dict = {}
        sample_dict['symbol'] = ticker

        bid_net = uniform(-1, 1)
        ask_net = uniform(bid_net, 1)
        sample_dict['pctBidNet'] = bid_net
        sample_dict['pctAskNet'] = ask_net
        sample_dict['bid'] = 0
        sample_dict['ask'] = 0
        sample_dict['bidVenue'] = 'NSDQ'
        sample_dict['askVenue'] = 'NSDQ'

        return sample_dict
