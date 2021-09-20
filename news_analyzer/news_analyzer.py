from typing import Tuple
import numpy as np
from news_analyzer.preprocessing import preprocess
import spacy

RELEVANT_NEWS_WORDS = ['upgrade', 'neutral', 'downgrade',
                       'overweight', 'equalweight', 'underweight',
                       'raised', 'lowered', 'Q1', 'Q2', 'Q3', 'Q4',
                       'reports', 'guides', 'report', 'guidance',
                       'merge', 'acquire', 'acquisition',
                       'initiated', 'sees', 'maintain', 'outlook', 'target']


class NewsAnalyzer:
    def __init__(self):
        self.__nlp = spacy.load('en_core_web_sm')
        self.relevant_words_embeddings, self.relevant_words_embeddings_matrix =\
            self.__init_embeddings()

    def __init_embeddings(self) -> Tuple[dict, np.ndarray]:
        embeddings = {}
        relevant_words_str = ' '.join(RELEVANT_NEWS_WORDS)
        doc = self.__nlp(relevant_words_str)
        for token in doc:
            print(token)
            vec = token.vector / token.vector_norm  # l2 normalization
            print(vec)
            embeddings[str(token)] = vec
        embeddings_matrix = np.stack(embeddings.values())
        return embeddings, embeddings_matrix

    def get_relevance(self, text: str) -> int:
        text = preprocess(text)
        print(text)
        doc = self.__nlp(text)
        embeddings_array = doc[0].vector / doc[0].vector_norm
        for token in doc[1:]:
            print(token)
            vec = token.vector / token.vector_norm
            print(vec)
            embeddings_array = np.vstack((embeddings_array, vec))
        relevance_matrix = np.dot(self.relevant_words_embeddings_matrix, embeddings_array.T) > 0.8
        print(True in relevance_matrix)

        return 0


na = NewsAnalyzer()
# news = 'Nvidia price target raised to $275 from $260 at BofA'
# not_news = 'Yesterday Apple rose for more than 1%'
# na.relevant_words_embeddings['raised']
# na.get_relevance(text=news)
