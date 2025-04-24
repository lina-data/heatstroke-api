from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """
        columns: قائمة بالأعمدة التي سيتم معالجة القيم الشاذة فيها.
                 إذا لم تُحدد، سيتم تطبيق المعالجة على جميع الأعمدة.
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols_to_process = self.columns if self.columns is not None else X.columns

        for col in cols_to_process:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            median = X[col].median()
            X[col] = X[col].apply(lambda x: median if x < lower or x > upper else x)

        return X
