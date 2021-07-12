import numpy as np
from sklearn.linear_model import RidgeCV

class InferenceModule:
    def __init__(self):
        self.__model = RidgeCV(fit_intercept=False)
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) :
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
