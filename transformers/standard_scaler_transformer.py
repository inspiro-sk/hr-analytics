import numpy as np
from sklearn.preprocessing import StandardScaler


class StandardScalerTransformer:
    def __init__(self, data, cols):
        self.transformer = None

    def fit(self, data: np.ndarray, cols: list = None):
        # get indices of columns from numpy array that should be scaled
        cols_idx = [data.columns.index(col) for col in cols]

        data = data[:, cols_idx]

        self.transformer = StandardScaler(copy=False)
        self.transformer.fit(data)

        return self.transformer

    def transform(self, data: np.ndarray, cols: list):
        cols_idx = [data.columns.index(col) for col in cols]

        data = data[:, cols_idx]
        transform_data = self.transformer.transform(data)
        return transform_data
