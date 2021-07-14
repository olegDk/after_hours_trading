import pickle
import os
import messages
from random import choices, uniform


def get_tickers(path: str) -> dict:
    list_subfolders_with_paths =\
        [f.path for f in os.scandir(path) if f.is_dir()]

    tickers = []

    with open(f'{path}/all_etfs.pkl', "rb") as input:
        f_list = pickle.load(input)
        tickers = tickers + f_list

    for sector_dir in list_subfolders_with_paths:
        for f in os.listdir(sector_dir):
            with open(f'{sector_dir}/{f}', "rb") as input:
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
        self.__tickers_dict = get_tickers(path='modeling/tickers')

    def sample_l1_update(self) -> dict:
        sample_dict = messages.market_data()

        # Sample ticker, %BidNet and %AskNet
        tickers = list(self.__tickers_dict.keys())
        probs = list(self.__tickers_dict.values())

        ticker = choices(tickers, probs)[0]
        bid_net = uniform(-1, 1)
        ask_net = uniform(bid_net, 1)

        sample_dict['symbol'] = ticker
        sample_dict['%bidNet'] = bid_net
        sample_dict['%askNet'] = ask_net

        return sample_dict
