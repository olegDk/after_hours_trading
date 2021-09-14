import pickle
import sys
import os
from copy import deepcopy
from config import messages
from typing import Tuple
from config.constants import *
from random import uniform, randint
from datetime import datetime
import numpy as np

RELEVANT_NEWS_WORDS = ['upgrade', 'neutral', 'downgrade',
                       'overweight', 'equalweight', 'underweight',
                       'raised', 'lowered', 'Q1', 'Q2', 'Q3', 'Q4',
                       'reports', 'guides', 'report', 'guidance',
                       'merge', 'acquire', 'acquisition',
                       'initiated', 'sees']

POSITIVE_WORDS = ['upgrade', 'overweight', 'raised',
                  'merge', 'acquire', 'acquisition']

NEGATIVE_WORDS = ['downgrade', 'underweight', 'lowered']

NEUTRAL_WORDS = ['neutral', 'equalweight', 'Q1', 'Q2', 'Q3', 'Q4',
                 'reports', 'guides', 'report', 'guidance',
                 'initiated', 'sees']

WORDS_DICT = {
    'POSITIVE': POSITIVE_WORDS,
    'NEGATIVE': NEGATIVE_WORDS,
    'NEUTRAL': NEUTRAL
}

assert len(RELEVANT_NEWS_WORDS) == \
       len(POSITIVE_WORDS) + len(NEGATIVE_WORDS) + len(NEUTRAL_WORDS)


def get_tickers() -> dict:
    if sys.gettrace():
        cwd = os.getcwd()
        path = f'{cwd}/../analytics/modeling'
    else:
        path = f'analytics/modeling'
    list_subfolders_with_paths =\
        [f.path for f in os.scandir(f'{path}/sectors') if f.is_dir()]

    tickers = []
    with open(f'{path}/all_indicators.pkl', "rb") as i:
        f_list = pickle.load(i)
        tickers = tickers + f_list

    for sector_dir in list_subfolders_with_paths:
        for f in os.listdir(f'{sector_dir}/tickers'):
            with open(f'{sector_dir}/tickers/{f}', "rb") as i:
                if f.startswith('traidable_tickers'):
                    f_list = pickle.load(i)
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


def get_words() -> Tuple[dict, dict, dict, dict]:
    if sys.gettrace():
        cwd = os.getcwd()
        path = f'{cwd}/../emulator/words.pkl'
    else:
        path = f'emulator/words.pkl'

    with open(path, 'rb') as i:
        words = pickle.load(i)

    # Manually approximate sum of probabilities for most relevant words
    sum_relevant = 0.6
    sum_default = float(1-sum_relevant)
    sum_positive = sum_negative = sum_neutral = float(sum_relevant/3)

    # Uniformly distribute probability mass across rest of the tickers
    uniform_positive_prob =\
        float(sum_positive / (len(POSITIVE_WORDS)))
    uniform_negative_prob = \
        float(sum_negative / (len(NEGATIVE_WORDS)))
    uniform_neutral_prob = \
        float(sum_neutral / (len(NEUTRAL_WORDS)))
    uniform_default_prob = \
        float(sum_default / (len(words) - len(RELEVANT_NEWS_WORDS)))

    default_words = list(set(words) - set(RELEVANT_NEWS_WORDS))

    uniform_default_dist = [uniform_default_prob for _ in default_words]

    prob_default_dict = dict(zip(default_words, uniform_default_dist))

    uniform_positive_dist = [uniform_positive_prob for _ in POSITIVE_WORDS]

    prob_positive_dict = dict(zip(POSITIVE_WORDS, uniform_positive_dist))

    uniform_negative_dist = [uniform_negative_prob for _ in NEGATIVE_WORDS]

    prob_negative_dict = dict(zip(NEGATIVE_WORDS, uniform_negative_dist))

    uniform_neutral_dist = [uniform_neutral_prob for _ in NEUTRAL_WORDS]

    prob_neutral_dict = dict(zip(NEUTRAL_WORDS, uniform_neutral_dist))

    return prob_default_dict, prob_positive_dict,\
           prob_negative_dict, prob_neutral_dict


def get_uniform_tickers() -> dict:
    if sys.gettrace():
        cwd = os.getcwd()
        path = f'{cwd}/../analytics/modeling'
    else:
        path = f'analytics/modeling'
    list_subfolders_with_paths =\
        [f.path for f in os.scandir(f'{path}/sectors') if f.is_dir()]

    tickers = []

    with open(f'{path}/all_indicators.pkl', "rb") as i:
        f_list = pickle.load(i)
        tickers = tickers + f_list

    for sector_dir in list_subfolders_with_paths:
        for f in os.listdir(f'{sector_dir}/tickers'):
            with open(f'{sector_dir}/tickers/{f}', "rb") as i:
                if f.startswith('traidable_tickers'):
                    f_list = pickle.load(i)
                    tickers = tickers + f_list

    # Removing duplicates
    tickers = list(set(tickers))
    uniform_prob = float(1/len(tickers))
    uniform_dist = [uniform_prob for _ in tickers]
    prob_dict = dict(zip(tickers, uniform_dist))

    return prob_dict


def sample_stock_l1(ticker: str) -> dict:
    close = round(np.random.uniform(990.0, 1010.0), 2)
    sample_dict = {SYMBOL: ticker, CLOSE: close}
    bid_net = uniform(-1, 1)
    ask_net = uniform(bid_net, 1)
    sample_dict[PCT_BID_NET] = bid_net
    sample_dict[PCT_ASK_NET] = ask_net
    sample_dict[BID] = 1000.0
    sample_dict[ASK] = 1001.5
    sample_dict[BID_VENUE] = 1
    sample_dict[ASK_VENUE] = 1
    return sample_dict


def sample_stock_news(ticker: str,
                      words_dict: dict,
                      num_words: int) -> dict:
    sample_dict = {SYMBOL: ticker}
    words = list(words_dict.keys())
    len_words = len(words)
    probs = [float(1/len_words) for _ in words]
    content_list = np.random.choice(a=words,
                                    size=num_words,
                                    replace=True,
                                    p=probs)
    content = ' '.join(content_list)
    sample_dict[CONTENT] = content
    sample_dict[RELEVANCE] = 9
    sample_dict[AMC_KEY] = 0
    sample_dict[DATETIME] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return sample_dict


class MarketDataGenerator:
    def __init__(self):
        self.__tickers_dict = get_tickers()
        self.__tickers_uniform_dict = get_uniform_tickers()
        self.__default_dict, self.__positive_dict,\
            self.__negative_dict, self.__neutral_dict = get_words()
        self.__sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        self.__sentiment_probs = [1 / 3 for _ in range(3)]
        self.__sentiment_to_dict = {
            'POSITIVE': deepcopy(self.__positive_dict),
            'NEGATIVE': deepcopy(self.__negative_dict),
            'NEUTRAL': deepcopy(self.__neutral_dict)
        }

    def sample_acc_snapshot(self) -> dict:
        data_dict = {}
        num_positions = randint(a=1, b=10)
        num_orders = randint(a=1, b=10)
        # Sample ticker, pctBidNet and pctAskNet
        tickers = list(self.__tickers_uniform_dict.keys())
        probs = list(self.__tickers_uniform_dict.values())
        sampled_pos_tickers = list(np.random.choice(tickers,
                                                    size=num_positions,
                                                    replace=False,
                                                    p=probs))
        side_list = [BUY, SELL]
        side_probs = [0.5, 0.5]
        pos_list = []
        orders_list = []
        for ticker in sampled_pos_tickers:
            pos_dict = {SYMBOL: ticker}
            side = np.random.choice(side_list,
                                    size=1,
                                    replace=False,
                                    p=side_probs)
            pos_dict[SIDE] = side[0]
            pos_dict[SIZE] = randint(a=1, b=100)
            pos_dict[PRICE] = np.random.uniform(990.0, 1010.0)
            pos_list.append(pos_dict)

        sampled_orders_tickers = list(np.random.choice(tickers,
                                                       size=num_orders,
                                                       replace=False,
                                                       p=probs))
        for ticker in sampled_orders_tickers + sampled_pos_tickers:
            orders_dict = {SYMBOL: ticker}
            side = np.random.choice(side_list,
                                    size=1,
                                    replace=False,
                                    p=side_probs)
            orders_dict[SIDE] = side[0]
            orders_dict[SIZE] = randint(a=1, b=100)
            orders_dict[PRICE] = np.random.uniform(990.0, 1010.0)
            orders_list.append(orders_dict)

        data_dict[ORDERS] = orders_list
        data_dict[POSITIONS] = pos_list

        return data_dict

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

        sampled_l1_dicts = list(map(sample_stock_l1, sampled_tickers))
        sample_dict[DATA] = sampled_l1_dicts

        return sample_dict

    def sample_news_update(self) -> dict:
        sample_dict = messages.news_data()
        # Sample ticker, pctBidNet and pctAskNet
        tickers = list(self.__tickers_uniform_dict.keys())
        probs = list(self.__tickers_uniform_dict.values())

        n_tickers = randint(1, 6)
        sampled_tickers = np.random.choice(tickers,
                                           size=n_tickers,
                                           replace=False,
                                           p=probs)

        sampled_news_dicts = []
        for ticker in sampled_tickers:
            sentiment = np.random.choice(a=self.__sentiments,
                                         size=1,
                                         replace=False,
                                         p=self.__sentiment_probs)[0]
            words_dict = self.__sentiment_to_dict[sentiment]
            n_words = randint(a=1, b=10)
            sample_news_dict = sample_stock_news(ticker, words_dict, n_words)
            sampled_news_dicts = sampled_news_dicts + [sample_news_dict]

        sample_dict[DATA] = sampled_news_dicts
        return sample_dict


md = MarketDataGenerator()
md.sample_news_update()
