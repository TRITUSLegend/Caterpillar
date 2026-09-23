"""
Shared preprocessing used inside the saved model pipelines.

The anomaly model pickle references PerTypeScaler by this module path
(`preprocessing.PerTypeScaler`), so anything that loads the .pkl files must be
able to `import preprocessing` - keep this file name and location stable.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Last 2 hours of shift A (06:00-14:00) and shift B (14:00-22:00), UTC
END_OF_SHIFT_HOURS = (12, 13, 20, 21)


class PerTypeScaler(BaseEstimator, TransformerMixin):
    """Z-normalizes numeric features within each machine type.

    Fuel use and load cycles differ a lot between machine types, so a reading is
    compared against machines of the same type rather than the whole fleet.

    Features listed in `end_of_shift_features` are additionally split by whether
    the reading falls in the last 2 hours of a shift (from `time_col`), so the
    normal end-of-shift idle bursts are compared against each other instead of
    looking like anomalies.
    """

    def __init__(self, group_col="Machine_Type", features=None, time_col=None, end_of_shift_features=()):
        self.group_col = group_col
        self.features = features
        self.time_col = time_col
        self.end_of_shift_features = end_of_shift_features

    def _keys(self, X, feature):
        if feature in self.end_of_shift_features:
            hours = pd.to_datetime(X[self.time_col], utc=True).dt.hour
            return [X[self.group_col], hours.isin(END_OF_SHIFT_HOURS).rename("End_Of_Shift")]
        return X[self.group_col]

    def fit(self, X, y=None):
        self.groups_ = set(X[self.group_col])
        self.stats_ = {}
        for f in self.features:
            grouped = X[f].groupby(self._keys(X, f))
            std = grouped.std(ddof=0).replace(0, 1.0)
            self.stats_[f] = (grouped.mean().to_dict(), std.to_dict())
        return self

    def transform(self, X):
        unknown = set(X[self.group_col]) - self.groups_
        if unknown:
            raise ValueError(f"Unknown {self.group_col}: {sorted(unknown)}")

        out = pd.DataFrame(index=X.index, columns=self.features, dtype=float)
        for f in self.features:
            means, stds = self.stats_[f]
            for key, rows in X[f].groupby(self._keys(X, f)):
                out.loc[rows.index, f] = (rows - means[key]) / stds[key]
        return out.to_numpy()
