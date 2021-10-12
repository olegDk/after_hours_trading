import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

pd.options.mode.chained_assignment = None


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
        print(f'numeric_features: {numeric_features}')
        print(f'categorical_features: {categorical_features}')
        return {'numeric_features': numeric_features, 'categorical_features': categorical_features}
