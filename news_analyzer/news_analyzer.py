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
                       'reiterate', 'rating']


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
                                  embeddings_matrix.T) > 0.5

        return True in relevance_matrix
