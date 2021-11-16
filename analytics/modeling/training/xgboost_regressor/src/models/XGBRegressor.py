from abc import ABCMeta, abstractmethod
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeCV
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)


class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame = None, y: np.ndarray = None):
        """
        Fitting preprocessing pipeline
        :param X: features dataframe
        :param y: labels np.ndarray
        """

        feature_types = Preprocessor.identify_type(X)
        numeric_features = feature_types['numeric_features']
        categorical_features = feature_types['categorical_features']
        X[categorical_features] = X[categorical_features].apply(lambda x: x.astype(str))

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=-9999))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        return self.preprocessor.fit(X)

    def transform(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        Transform new data using preprocessing pipeline
        :param X: features dataframe
        :return: transformed features dataframe
        """

        return self.preprocessor.transform(X)

    @staticmethod
    def identify_type(X: pd.DataFrame) -> dict:
        """
        Correctly identify features as numeric or categorical
        not only by types in dataframe, but also by counting the number
        of unique values for numeric features with categorical threshold.
        (If number of unique values is less than categorical threshold,
        then numeric feature will be considered as categorical).

        :param X: features dataframe
        :return: dict with names of numeric and categorical features
        """
        numeric_mask = X.dtypes != object
        numeric_features = X.columns[numeric_mask].tolist()
        categorical_features = X.columns[~numeric_mask].tolist()
        categorical_threshold = 50
        for feature in numeric_features:
            if len(X[feature].unique().tolist()) < categorical_threshold:
                categorical_features.append(feature)
        numeric_features = list(set(X.columns.tolist()) - set(categorical_features))
        assert(len(X.columns.tolist()) == (len(categorical_features) + len(numeric_features)))
        return {'numeric_features': numeric_features, 'categorical_features': categorical_features}


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
            'regressor__learning_rate': [0.01, 1],
            'regressor__gamma': [0.01, 1],
            'regressor__reg_lambda': [0.01, 1],
            'regressor__max_depth': [1, 5],
            'regressor__subsample': [0.5, 1],
            'regressor__colsample_bytree': [0.5, 1]
        }
        opt = RandomizedSearchCV(self.make_pipeline(), search_spaces, n_iter=5, cv=5,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        opt.fit(X, y)
        scores = {}
        best_params = dict(opt.best_params_)
        best_score = opt.best_score_
        best_estimator = opt.best_estimator_
        cv_result = cross_val_score(estimator=best_estimator,
                                    X=X, y=y,
                                    scoring='neg_mean_squared_error',
                                    cv=5)
        scores['mae'] = -best_score
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


# xgbr = XGBRegressor()
# data = pd.read_csv('/home/oleh/takion_trader/'
#                    'analytics/modeling/training/xgboost_regressor/data/data.csv',
#                    na_values='na')
# X = data.drop(['%Gap_CRM'], axis=1)
# y = data['%Gap_CRM']
#
# start = datetime.now()
# xgbr.fit(X, y)
# finish = datetime.now()
# xgbr_training_time = (finish - start).microseconds/1000
# print(f'XGBR training time: {xgbr_training_time}')
