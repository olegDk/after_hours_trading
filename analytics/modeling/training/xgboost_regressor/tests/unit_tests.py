import os
import unittest
from analytics.modeling.training.xgboost_regressor.src.models.XGBRegressor import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pathlib import Path
import pandas as pd
import numpy as np


class TestStringMethods(unittest.TestCase):

    def setUp(self) -> None:
        # Modify
        data_path = Path(
            f'{os.getcwd()}'
            f'/../'
            f'data/'
            f'data.csv'
        )
        data = pd.read_csv(data_path, na_values='na')
        self.X = data.drop(['%Gap_CRM'], axis=1)
        self.y = data['%Gap_CRM']

    def test_handle_missing_values(self):
        """
        In order to test missing_values handling we will impute
        numeric and categorical features separately as it is done
        in Preprocessor's main pipeline and compare results with expected output
        using simplified example inputs and expected outputs.
        We can pretend that scikit-learn's SimpleImputer works well, but
        let's doublecheck that.
        """
        si_numeric = SimpleImputer(strategy='constant', fill_value=-9999)
        si_categorical = SimpleImputer(strategy='constant', fill_value='missing')

        # Testing numeric imputation
        example_numeric_input = pd.DataFrame(
            [{'%Gap_QQQ': np.nan, '%Gap_XLK': 2.5, '%Gap_DIA': np.nan,
              '%Gap_XLF': 0.9, '%Gap_SPY': 1.2, '%Gap_TLT': 0.2},
             {'%Gap_QQQ': 0.9, '%Gap_XLK': np.nan, '%Gap_DIA': 0.5,
              '%Gap_XLF': 0.9, '%Gap_SPY': 1.2, '%Gap_TLT': 0.2}])
        expected_numeric_output = np.array([[-9999, 2.5, -9999, 0.9, 1.2, 0.2],
                                            [0.9, -9999, 0.5, 0.9, 1.2, 0.2]])
        numeric_transform_result = si_numeric.fit_transform(example_numeric_input)
        self.assertTrue((expected_numeric_output == numeric_transform_result).all())

        # Categorical imputation
        example_categorical_input = pd.DataFrame(
            [{'NearSigmaFlag_CRM': 'Standard_range', 'NearBollingerBandsFlag_CRM': 'NearSMA20LowerSigma',
              'NearSigmaFlag_QQQ': np.nan, 'NearBollingerBandsFlag_QQQ': 'NearSMA20LowerSigma'},
             {'NearSigmaFlag_CRM': 'Standard_range', 'NearBollingerBandsFlag_CRM': np.nan,
              'NearSigmaFlag_QQQ': np.nan, 'NearBollingerBandsFlag_QQQ': 'NearSMA20LowerSigma'}])
        expected_categorical_output = np.array([['Standard_range', 'NearSMA20LowerSigma',
                                                 'missing', 'NearSMA20LowerSigma'],
                                                ['Standard_range', 'missing',
                                                 'missing', 'NearSMA20LowerSigma']], dtype=object)
        categorical_transform_result = si_categorical.fit_transform(example_categorical_input)
        self.assertTrue((expected_categorical_output == categorical_transform_result).all())

    def test_new_categories_handling(self):
        """
        Testing new categories handling by testing
        whether adding new category results in expected Preprocessors transform type
        and whether fitted XGBEstimator pipeline will predict without any errors
        """
        # Adding SomeNewCategory to home_ownership and verification_status columns
        example_input = pd.DataFrame([{'%Gap_QQQ': 2, '%Gap_XLK': 2.5, '%Gap_DIA': 1,
                                       '%Gap_XLF': 0.9, '%Gap_SPY': 1.2, '%Gap_TLT': 1.5,
                                       'NearSigmaFlag_CRM': 'Standard_range',
                                       'NearBollingerBandsFlag_CRM': 'NearSMA20LowerSigma',
                                       'NearSigmaFlag_QQQ': 'Some_new_category',
                                       'NearBollingerBandsFlag_QQQ': 'NearSMA20LowerSigma'
                                       }])

        # Fitting XGBEstimator pipeline
        xgbr = XGBRegressor()
        xgbr.fit(self.X, self.y)
        prediction = xgbr.predict(example_input)
        self.assertIsInstance(prediction, np.ndarray)

    def test_return_types(self):
        """
        Simple output types testing
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=True)

        xgbe = XGBRegressor()
        fit_result = xgbe.fit(self.X, self.y)
        self.assertIsNone(fit_result)

        predict_result = xgbe.predict(X_test)
        self.assertIsInstance(predict_result, np.ndarray)

        evaluate_result = xgbe.evaluate(self.X, self.y)
        self.assertIsInstance(evaluate_result, dict)
        self.assertIn('mae', list(evaluate_result.keys()))
        self.assertIn('mse', list(evaluate_result.keys()))

        tune_result = xgbe.tune(self.X, self.y)
        self.assertIsInstance(tune_result, dict),
        self.assertIn('best_params', list(tune_result.keys()))
        self.assertIn('best_scores', list(tune_result.keys()))


if __name__ == '__main__':
    unittest.main()
