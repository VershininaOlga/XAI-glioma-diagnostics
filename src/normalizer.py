import pickle
import pandas as pd
import numpy as np


class normalizer:
    def __init__(self):
        with open('models/zscaler.pkl', 'rb') as file:
            self.zscaler = pickle.load(file)

    def prepare_data(self, X_raw):
        X = X_raw[self.zscaler['features'].values]
        X = pd.DataFrame(np.log2(X.values + 1), index=X.index, columns=X.columns)
        means = self.zscaler['means'].values
        stds = self.zscaler['stds'].values
        X = (X - means) / stds
        if np.any(X.isna()):
            X = X.dropna()
            print('Samples containing missing values (NaNs) ​​were removed!')
        return X
