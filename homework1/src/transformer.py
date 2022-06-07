import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_norm):
        self.column_to_norm = column_to_norm

    def fit(self, features, label=None):
        return self

    def transform(self, features, label=None):
        features_ = features.copy()
        column = features_[self.column_to_norm]
        features_[self.column_to_norm] = (column - column.min()) / (column.max() - column.min())
        return features_
