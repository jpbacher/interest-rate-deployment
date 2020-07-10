from sklearn.base import BaseEstimator, TransformerMixin


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""
    def __init__(self, features=None):
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X, y):
        self.imputer_dict = {}
        for feat in self.features:
            self.imputer_dict[feat] = X[feat].median()
        return self

    def transform(self, X):
        X = X.copy()
        for feat in self.features:
            X[feat].fillna(self.imputer_dict[feat], inplace=True)
        return X
