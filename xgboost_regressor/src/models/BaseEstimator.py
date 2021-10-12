from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np


class BaseEstimator(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit on training data

        :param X: features dataframe
        :param y: labels array
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict on given features data

        :param X: features dataframe
        :return: predictions np.ndarray
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """
        Evaluate model

        :param X: features dataframe
        :param y: labels np.ndarray
        :return: dict with values of f1_score and logloss
        """
        raise NotImplementedError()

    @abstractmethod
    def tune(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """
        Find best hyperparameters using stratified K-Fold cross-validation for evaluation

        :param X: features dataframe
        :param y: labels np.ndarray
        :return: dict with best hyperparameters and cv scores for f1 and logloss
        """
        raise NotImplementedError()
