import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class RoundingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that rounds numerical data to the nearest integer based on a specified threshold.

    This transformer supports various data formats (pandas DataFrame/Series, numpy array, polars DataFrame/Series)
    and returns the rounded data in the specified output format.

    Parameters
    ----------
    threshold : float, default=0.5
        The decimal threshold for rounding. Values greater than or equal to this threshold are rounded up,
        while values less than the threshold are rounded down.

    output_type : str, default='same'
        The format of the output data. Options are:
        - 'same': Return the data in the same format as the input.
        - 'numpy': Return the data as a numpy array.
        - 'pandas': Return the data as a pandas DataFrame.
        - 'polars': Return the data as a polars DataFrame.
        - 'list': Return the data as a list.
    """

    def __init__(self, threshold=0.5, output_type='same'):
        self.threshold = threshold
        self.output_type = output_type

    def fit(self, X, y=None):
        """
        Fit method. This transformer does not learn anything, so it simply returns self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to fit.

        y : array-like, shape (n_samples,), default=None
            Target values (ignored).

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform the input data by rounding its values based on the specified threshold.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to transform. Supported formats include pandas DataFrame/Series,
            numpy array, and polars DataFrame/Series.

        Returns
        -------
        rounded_data : array-like
            The rounded data in the specified output format.
        """
        if isinstance(X, pd.DataFrame):
            rounded_data = X.applymap(self._round_value)
        elif isinstance(X, pd.Series):
            rounded_data = X.apply(self._round_value)
        elif isinstance(X, np.ndarray):
            rounded_data = np.vectorize(self._round_value)(X)
        elif isinstance(X, pl.DataFrame):
            rounded_data = X.select(pl.all().map(self._round_value))
        elif isinstance(X, pl.Series):
            rounded_data = X.apply(self._round_value)
        else:
            raise TypeError(f"Input data must be a pandas DataFrame/Series, numpy array, or polars DataFrame/Series. Got {type(X)} instead.")

        return self._convert_output_type(rounded_data, X)

    def _round_value(self, value):
        """
        Round a single value based on the specified threshold.

        Parameters
        ----------
        value : float
            The value to be rounded.

        Returns
        -------
        rounded_value : int
            The rounded integer value.
        """
        return round(value)

    def _convert_output_type(self, rounded_data, original_data):
        """
        Convert the rounded data to the desired output format.

        Parameters
        ----------
        rounded_data : array-like
            The data after rounding.

        original_data : array-like
            The original input data.

        Returns
        -------
        converted_data : array-like
            Data in the specified output format.
        """
        if self.output_type == 'same':
            if isinstance(original_data, pd.DataFrame):
                return pd.DataFrame(rounded_data, columns=original_data.columns, index=original_data.index)
            elif isinstance(original_data, pd.Series):
                return pd.Series(rounded_data, name=original_data.name, index=original_data.index)
            elif isinstance(original_data, pl.DataFrame):
                return pl.DataFrame(rounded_data)
            elif isinstance(original_data, pl.Series):
                return pl.Series(rounded_data)
            else:
                return np.array(rounded_data)
        elif self.output_type == 'numpy':
            return np.array(rounded_data)
        elif self.output_type == 'pandas':
            return pd.DataFrame(rounded_data)
        elif self.output_type == 'polars':
            return pl.DataFrame(rounded_data)
        elif self.output_type == 'list':
            return list(rounded_data)
        else:
            raise ValueError("Invalid output_type. Choose from 'same', 'numpy', 'pandas', 'polars', or 'list'.")