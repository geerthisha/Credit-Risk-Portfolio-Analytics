import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class UpperQuantileCapper(BaseEstimator, TransformerMixin):
    """Learn upper quantile caps for selected feature positions."""

    def __init__(self, column_indices=(), quantile=0.99):
        self.column_indices = column_indices
        self.quantile = quantile

    def fit(self, X, y=None):
        if not 0 < self.quantile <= 1:
            raise ValueError("quantile must be in (0, 1].")

        values = check_array(X, dtype=float)

        self.n_features_in_ = values.shape[1]
        self.caps_ = np.full(self.n_features_in_, np.inf)

        for index in self.column_indices:
            if not 0 <= index < self.n_features_in_:
                raise ValueError("Invalid capping column index.")

            self.caps_[index] = np.quantile(
                values[:, index],
                self.quantile,
            )

        return self

    def transform(self, X):
        check_is_fitted(self, "caps_")
        values = check_array(X, dtype=float)

        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of features has changed.")

        return np.minimum(values, self.caps_)


class IncomeLogTransformer(BaseEstimator, TransformerMixin):
    """Optionally apply log1p to the selected income column."""

    def __init__(self, column_index, enabled=False):
        self.column_index = column_index
        self.enabled = enabled

    def fit(self, X, y=None):
        values = check_array(X, dtype=float)
        self.n_features_in_ = values.shape[1]

        if not 0 <= self.column_index < self.n_features_in_:
            raise ValueError("Invalid income column index.")

        if self.enabled and (values[:, self.column_index] < 0).any():
            raise ValueError("Income must be non-negative for log1p.")

        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = check_array(X, dtype=float).copy()

        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of features has changed.")

        if self.enabled:
            if (values[:, self.column_index] < 0).any():
                raise ValueError("Income must be non-negative for log1p.")

            values[:, self.column_index] = np.log1p(
                values[:, self.column_index]
            )

        return values