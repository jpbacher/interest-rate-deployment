import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class StandardizeFeatures:
    """
    Standardizes numerical features to zero mean, unit variance, & keeps column names
    of dataframe (StandardScalar returns a numpy array).
    """
    def __init__(self, cols_to_std):
        self.cols_to_std = cols_to_std
        self.sc = StandardScaler()
        self._is_fit = False

    def fit(self, X, y=None):
        self._is_fit = True
        self.sc.fit(X[self.cols_to_std])
        return self

    def transform(self, X, y=None):
        if not self._is_fit:
            raise FittedError('Features have not been fitted.')
        X_std_cols = pd.DataFrame(data=self.sc.transform(X[self.cols_to_std]),
                                  columns=self.cols_to_std, index=X.index)
        X_new = X.drop(self.cols_to_std, axis=1)
        X_new = pd.concat([X_new, X_std_cols], axis=1)
        return X_new

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class NormalizeFeatures:
    """
    Transform features to a range between 0 & 1.
    """
    def __init__(self, cols_to_normalize):
        self.cols_to_normalize = cols_to_normalize
        self.norm = MinMaxScaler()
        self._is_fit = False

    def fit(self, X, y=None):
        self._is_fit = True
        self.norm.fit(X[self.cols_to_normalize])
        return self

    def transform(self, X, y=None):
        if not self._is_fit:
            raise FittedError('Features have not been fitted.')
        X_norm_cols = pd.DataFrame(data=self.norm.transform(X[self.cols_to_normalize]),
                                   columns=self.cols_to_normalize, index=X.index)
        X_new = X.drop(self.cols_to_normalize, axis=1)
        X_new = pd.concat([X_new, X_norm_cols], axis=1)
        return X_new

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)



class OneHotEncodeFeatures:
    """
    One-hot-encode categorical features; class assumes there does not exist missing values
    (we will not have a dummy na column).
    """
    def __init__(self):
        self._was_fit = False

    def fit(self, X, y=None):
        self._was_fit = True
        # retrieve list of nominal columns
        cols_to_encode = X.dtypes[(X.dtypes == 'object') | (X.dtypes == 'category')].index
        self.cat_cols = [col for col in cols_to_encode]
        dummy = pd.get_dummies(X, columns=self.cat_cols, dummy_na=False)
        self.col_names = dummy.columns
        return self

    def transform(self, X, y=None):
        if not self._was_fit:
            raise FittedError('Features have not been fitted.')
        X_new = pd.get_dummies(X, columns=self.cat_cols, dummy_na=False)
        # if new categories in test set that model has not seen, set to 0
        new_cols = set(self.col_names) - set(X_new.columns)
        for col in new_cols:
            X_new[col] = 0
        X_new = X_new[self.col_names]
        return X_new

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    

class PreprocessTitanic:
    def __init__(self, non_pred_feats, missing_feats):
        self.columns_to_drop = non_pred_feats + missing_feats
        self._was_fit = False
        
    def fit(self, X, y=None):
        self._was_fit = True
        X_new = X.drop(self.columns_to_drop, axis=1)
        self.impute_values = {'age': X['age'].median(),
                              'fare': X['fare'].median(),
                              'embarked': X['embarked'].value_counts().index[0]
                             }
        X_new.fillna(value=self.impute_values, inplace=True)
        cat_feats = X_new.dtypes[(X_new.dtypes == 'object') | 
                                 (X_new.dtypes == 'category')].index
        self.cat_feats = [feat for feat in cat_feats]
        dummy = pd.get_dummies(X_new, columns=self.cat_feats)
        self.feat_names = dummy.columns
        del dummy
        return self
    
    def transform(self, X, y=None):
        if not self._was_fit:
            raise FittedError('Fit method must be called before transforming.')
        X_new = X.drop(self.columns_to_drop, axis=1)
        X_new.fillna(value=self.impute_values, inplace=True)
        X_new = pd.get_dummies(X_new, columns=self.cat_feats)
        new_feats = set(self.feat_names) - set(X_new.columns)
        for feat in new_feats:
            X_new[feat] = 0
        X_new = X_new[self.feat_names]
        return X_new
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)