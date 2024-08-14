from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class InductiveWinsorizer(BaseEstimator, TransformerMixin):
    """
    A transformer that performs inductive winsorization on numerical data, clipping outliers
    based on specified percentile limits. This transformer can handle Pandas DataFrames, Series, and NumPy arrays.

    Parameters
    ----------
    limits : tuple, default=(0.05, 0.05)
        A tuple specifying the lower and upper percentile limits for winsorization.
        For example, (0.05, 0.05) will clip the bottom 5% and the top 5% of values.

    min_unique_values : int, default=10
        Minimum number of unique values required in a column to apply winsorization.
        If a column has fewer unique values than this threshold, winsorization is not applied.

    output_format : str, default='dataframe'
        Format of the output after transformation. Can be 'dataframe' to return a Pandas DataFrame
        or 'numpy' to return a NumPy array.

    Attributes
    ----------
    lower_bounds : dict
        A dictionary storing the calculated lower bound for each column.

    upper_bounds : dict
        A dictionary storing the calculated upper bound for each column.

    Methods
    -------
    fit(X, y=None)
        Computes the lower and upper bounds for winsorization based on the data in X.

    transform(X)
        Applies the winsorization to the data in X, clipping values outside the computed bounds.

    fit_transform(X, y=None)
        Fits the transformer to X and returns the transformed data.

    _is_numerical(series)
        Checks if the provided Pandas Series is of a numerical data type.

    _convert_to_dataframe(X)
        Converts the input X into a Pandas DataFrame, if it is not already one.

    _convert_output_format(X)
        Converts the transformed DataFrame into the specified output format (DataFrame or NumPy array).
    """

    def __init__(self, limits=(0.05, 0.05), min_unique_values=10, output_format='dataframe'):
        self.limits = limits
        self.min_unique_values = min_unique_values
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.output_format = output_format

        self._validate_params()

    def _validate_params(self):
        if not isinstance(self.limits, tuple) or len(self.limits) != 2:
            raise ValueError("limits should be a tuple of two values.")
        if not (0 <= self.limits[0] < 1 and 0 <= self.limits[1] < 1):
            raise ValueError("limits values should be between 0 and 1.")
        if self.output_format not in ['dataframe', 'numpy']:
            raise ValueError("Output format not supported. Use 'dataframe' or 'numpy'.")

    def fit(self, X, y=None):
        """
        Fits the winsorizer to the data in X by computing the lower and upper bounds
        for each numerical column that meets the unique values threshold.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            The input data to fit the winsorizer. This can be a Pandas DataFrame, Series, or NumPy array.

        y : Ignored
            Not used, present here for consistency with the scikit-learn API.

        Returns
        -------
        self : InductiveWinsorizer
            The fitted instance of the transformer.
        """
        X = self._convert_to_dataframe(X)
        for column in X.columns:
            if self._is_numerical(X[column]) and X[column].nunique() >= self.min_unique_values:
                sorted_values = np.sort(X[column])
                n = len(sorted_values)
                self.lower_bounds[column] = sorted_values[int(self.limits[0] * n)]
                self.upper_bounds[column] = sorted_values[int((1 - self.limits[1]) * n)]
        return self

    def transform(self, X):
        """
        Transforms the data in X by clipping values outside the calculated bounds for each column.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            The input data to transform. This can be a Pandas DataFrame, Series, or NumPy array.

        Returns
        -------
        X_transformed : DataFrame or ndarray
            The transformed data, with outliers clipped based on the fitted bounds.
            The output format is determined by the 'output_format' parameter.
        """
        if not self.lower_bounds or not self.upper_bounds:
            raise RuntimeError("The transform method should only be called after the fit method.")
            
        X_copy = self._convert_to_dataframe(X)
        for column in X_copy.columns:
            if column in self.lower_bounds:
                X_copy[column] = np.clip(X_copy[column], self.lower_bounds[column], self.upper_bounds[column])
        return self._convert_output_format(X_copy)

    def fit_transform(self, X, y=None):
        """
        Fits the winsorizer to X and transforms the data in a single step.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            The input data to fit and transform. This can be a Pandas DataFrame, Series, or NumPy array.

        y : Ignored
            Not used, present here for consistency with the scikit-learn API.

        Returns
        -------
        X_transformed : DataFrame or ndarray
            The transformed data, with outliers clipped based on the fitted bounds.
            The output format is determined by the 'output_format' parameter.
        """
        return self.fit(X, y).transform(X)

    def _is_numerical(self, series):
        """
        Checks if a given Pandas Series is of a numerical data type.

        Parameters
        ----------
        series : Series
            The Pandas Series to check.

        Returns
        -------
        bool
            True if the series is numerical, False otherwise.
        """
        return pd.api.types.is_numeric_dtype(series)

    def _convert_to_dataframe(self, X):
        """
        Converts the input data into a Pandas DataFrame, if it is not already one.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            The input data to convert.

        Returns
        -------
        DataFrame
            The converted DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, pd.Series):
            return X.to_frame()
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        else:
            raise ValueError("Input type not supported. Expected DataFrame, Series, or NumPy array.")

    def _convert_output_format(self, X):
        """
        Converts the transformed DataFrame into the specified output format.

        Parameters
        ----------
        X : DataFrame
            The transformed data in DataFrame format.

        Returns
        -------
        DataFrame or ndarray
            The transformed data in the specified output format.
        """
        if self.output_format == 'dataframe':
            return X
        elif self.output_format == 'numpy':
            return X.to_numpy()
        else:
            raise ValueError("Output format not supported. Use 'dataframe' or 'numpy'.")