from xgboost_regressor.src.models.BaseEstimator import BaseEstimator
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost_regressor.src.preprocessing.Preprocessor import Preprocessor
import random

np.random.seed(42)
random.seed(42)


class XGBRegressor(BaseEstimator):

    def __init__(self):
        super().__init__()
        self.pipeline = self.make_pipeline()
        self.tune_pipeline = self.make_pipeline()

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit on training data

        :param X: features dataframe
        :param y: labels array
        :return: None
        """

        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict on given features data

        :param X: features dataframe
        :return: predictions np.ndarray
        """

        return self.pipeline.predict(X)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """
        Evaluate model

        :param X: features dataframe
        :param y: labels np.ndarray
        :return: dict with values of f1_score and logloss
        """

        scores = {}
        cv_result = cross_validate(estimator=self.pipeline, X=X, y=y,
                                   scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'], cv=5)
        scores['mae'] = -cv_result['test_neg_mean_absolute_error'].mean()
        scores['mse'] = -cv_result['test_neg_mean_squared_error'].mean()
        return scores

    def tune(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """
        Find best hyperparameters using stratified K-Fold cross-validation for evaluation

        :param X: features dataframe
        :param y: labels np.ndarray
        :return: dict with best hyperparameters and cv scores for f1 and logloss and best estimator refitted
        on whole train data
        """

        search_spaces = {
            'regressor__learning_rate': Real(0.01, 1),
            'regressor__gamma': Real(0.01, 1),
            'regressor__reg_lambda': Real(0.01, 1),
            'regressor__max_depth': Integer(1, 5),
            'regressor__subsample': Real(0.5, 1),
            'regressor__colsample_bytree': Real(0.5, 1)
        }
        opt = BayesSearchCV(self.make_pipeline(), search_spaces=search_spaces, n_iter=5, cv=5,
                            scoring='neg_mean_absolute_error')
        opt.fit(X, y)
        scores = {}
        best_params = dict(opt.best_params_)
        best_score = opt.best_score_
        best_estimator = opt.best_estimator_
        cv_result = cross_val_score(estimator=best_estimator,
                                    X=X, y=y,
                                    scoring='neg_mean_squared_error',
                                    cv=5)
        scores['mae'] = best_score
        scores['mse'] = -cv_result.mean()
        return {'best_params': best_params,
                'best_scores': scores,
                'best_estimator': best_estimator}

    def make_pipeline(self) -> Pipeline:
        pipeline = Pipeline(steps=[
            ('preprocessor', Preprocessor()),
            ('regressor', xgb.XGBRegressor(nthread=-1))
        ])
        return pipeline


xgbr = XGBRegressor()
