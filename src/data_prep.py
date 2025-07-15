import sys
import pickle
import pandas as pd
import numpy as np


class data_prep:
    def __init__(self):
        with open('models/clf.pkl', 'rb') as file:
            self.model = pickle.load(file)
            self.features = self.model['features'].astype(str).tolist()
        with open('models/zscaler.pkl', 'rb') as file:
            self.zscaler = pickle.load(file)

    
    def check(self, X_raw):
        X = X_raw.copy()
        is_full = set(self.features).issubset(list(X.columns))
        if not is_full:
            sys.exit(f'Error: The following columns are missing from the data: {set(self.features) - set(list(X.columns))}.\nThe model cannot be applied.')
        else:
            X = X[self.features]
            
        is_correct = X.dtypes.apply(lambda x: x.kind in {'i', 'f'}).all()
        if not is_correct:
            sys.exit(f'Error: The following columns are not numeric: {X.select_dtypes(exclude=["number"]).columns.tolist()}.\nThe model cannot be applied.')

        is_nan = np.any(X.isna())
        if is_nan:
            nan_indices = X.index[X.isna().any(axis=1)].to_list()
            X = X.dropna()
            print(f'Warning: Samples containing missing values (NaNs) ​​were removed: {nan_indices}')
            if len(X) == 0:
                sys.exit(f'Error: All samples contained missing values (NaNs) and were removed.\nThe model cannot be applied.')

        return X

    
    def znorm(self, X_raw):
        X = X_raw[self.features]
        X = pd.DataFrame(np.log2(X.values + 1), index=X.index, columns=X.columns)
        means = self.zscaler['means'].values
        stds = self.zscaler['stds'].values
        X = (X - means) / stds
        return X
