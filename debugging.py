import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for debugging during pipeline fitting by printing detailed information
    about the data at each step.

    This transformer is useful for understanding how data is transformed at each step in a pipeline,
    especially when working with complex data structures like lists of tuples or various data types.

    Parameters
    ----------
    step_name : str, default=""
        A label or name for the step in the pipeline. This name is printed with the debug information
        to identify the specific pipeline step.
    """

    def __init__(self, step_name=""):
        self.step_name = step_name

    def fit(self, X, y=None):
        """
        Fit method. This transformer does not learn anything, so it simply returns self.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit.

        y : array-like of shape (n_samples,), default=None
            Target values (ignored).

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform the input data by printing detailed debugging information about it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform and print debugging information about. Supported formats include
            lists, pandas DataFrame/Series, numpy arrays, and other array-like structures.

        Returns
        -------
        X : array-like
            The input data, unchanged.
        """
        print(f"Step: {self.step_name}")
        print(f"Input type: {type(X)}")

        if isinstance(X, list):
            print(f"List length: {len(X)}")
            if len(X) > 0:
                print(f"First element type: {type(X[0])}")
                if isinstance(X[0], tuple):
                    print(f"First tuple: {X[0]}")
                    print(f"First element of first tuple: {X[0][0]}")
                    print(f"Second element of first tuple: {X[0][1]}")
                    _, X = X[0]
                    print(f"After extraction:\n{X}")
                    print(f"First DataFrame shape: {X.shape}")
                    print(f"First DataFrame columns: {X.columns.tolist()}")
                    print(f"First DataFrame preview:\n{X.head()}")
                elif isinstance(X[0], pd.DataFrame):
                    print(f"First DataFrame shape: {X[0].shape}")
                    print(f"First DataFrame columns: {X[0].columns.tolist()}")
                    print(f"First DataFrame preview:\n{X[0].head()}")
                elif isinstance(X[0], np.ndarray):
                    print(f"First array shape: {X[0].shape}")
                    print(f"First array preview: {X[0]}")
                else:
                    print(f"First element: {X[0]}")
            else:
                print("List is empty.")
        elif isinstance(X, pd.DataFrame):
            print(f"Input shape: {X.shape}")
            print(f"Input columns: {X.columns.tolist()}")
            print(f"Data types:\n{X.dtypes}")
            print(f"Missing values:\n{X.isnull().sum()}")
            print(f"Summary statistics:\n{X.describe()}")
        elif isinstance(X, pd.Series):
            print(f"Input name: {X.name}")
            print(f"Data type: {X.dtype}")
            print(f"Missing values: {X.isnull().sum()}")
            print(f"Summary statistics:\n{X.describe()}")
        elif isinstance(X, np.ndarray):
            print(f"Input shape: {X.shape}")
            print(f"Data type: {X.dtype}")
            print(f"First few elements:\n{X[:5]}")
        else:
            print("Data type not recognized or unsupported for detailed debugging.")

        print("-" * 40)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the transformer and transform the data in a single step,
        while printing debugging information.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit and transform.

        y : array-like of shape (n_samples,), default=None
            Target values (ignored).

        **fit_params : dict
            Additional fitting parameters (ignored).

        Returns
        -------
        X : array-like
            The input data, unchanged.
        """
        return self.fit(X, y, **fit_params).transform(X)