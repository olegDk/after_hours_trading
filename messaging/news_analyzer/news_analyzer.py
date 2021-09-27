from typing import Tuple
import numpy as np
from news_analyzer.preprocessing import preprocess
import spacy

RELEVANT_NEWS_WORDS = ['upgrade', 'neutral', 'downgrade',
                       'overweight', 'equalweight', 'underweight',
                       'raised', 'lowered', 'Q1', 'Q2', 'Q3', 'Q4',
                       'reports', 'guides', 'report', 'guidance',
                       'merge', 'acquire', 'acquisition',
                       'initiated', 'sees', 'maintain', 'outlook', 'target',
                       'reiterate', 'rating', 'quarterly', 'dividend']


class NewsAnalyzer:
    def __init__(self):
        self.__nlp = spacy.load('en_core_web_lg')
        self.__relevant_words_embeddings_dict, self.__relevant_words_embeddings_matrix =\
            self.__init_embeddings()

    def __init_embeddings(self) -> Tuple[dict, np.ndarray]:
        tok_2_emb = {}
        word_doc = self.__nlp(RELEVANT_NEWS_WORDS[0])
        vec = word_doc.vector / word_doc.vector_norm
        embeddings_matrix = vec
        tok_2_emb[word_doc.text] = vec
        for relevant_word in RELEVANT_NEWS_WORDS[1:]:
            word_doc = self.__nlp(relevant_word)
            vec = word_doc.vector / word_doc.vector_norm
            tok_2_emb[word_doc.text] = vec
            embeddings_matrix = np.vstack((embeddings_matrix, vec))
        return tok_2_emb, embeddings_matrix

    def is_relevant(self, text: str) -> bool:
        text_list = preprocess(text).split(' ')
        word_doc = self.__nlp(text_list[0])
        vec = word_doc.vector / word_doc.vector_norm
        embeddings_matrix = vec
        for token in text_list[1:]:
            word_doc = self.__nlp(token)
            vec = word_doc.vector / word_doc.vector_norm
            embeddings_matrix = np.vstack((embeddings_matrix, vec))
        relevance_matrix = np.dot(self.__relevant_words_embeddings_matrix,
                                  embeddings_matrix.T) > 0.6

        return True in relevance_matrix


# na = NewsAnalyzer()
# not_news = '35 Stocks Moving In Mondays Mid-Day Session'
# not_news_1 = 'Form  8-K        OCCIDENTAL PETROLEUM'
# not_news_2 = 'Sector Briefing: Energy (Stock Price: 47.41 Change: -1.31)'
# not_news_3 = 'Leading And Lagging Sectors For September 20, 2021'
# not_news_4 = 'Citrons Andrew Left Coming Up On CNBCs Fast Money: Halftime Report'
# not_news_5 = 'Cathie Woods Ark Raises Stakes In Crypto Plays Robinhood, Coinbase -- And Other Keys Trades From Monday'
# not_news_6 = 'Cathie Woods ARK Invest Posts Fund Purchases For Friday,'
# news = 'Roth Capital Maintains Neutral on JinkoSolar Holding Co, Lowers Price Target to $51'
# news_1 = 'APA Corp. increases quarterly dividend to $0.0625/share, up from $0.025/share'
# na.is_relevant(text=not_news_6)
