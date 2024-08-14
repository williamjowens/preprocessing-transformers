import pandas as pd
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameExtractor(BaseEstimator, TransformerMixin):
    """
    A transformer that stores the data passed through the pipeline as an attribute, handling
    both DataFrames, NumPy arrays, and sparse matrices.

    This transformer is useful for debugging and understanding the intermediate steps
    in a machine learning pipeline. It stores the data during the `transform` step,
    allowing inspection of the data after the pipeline has been fitted.

    Parameters
    ----------
    feature_names : list, default=None
        A list of feature names to use if the input is a NumPy array or sparse matrix.
        If None and the input is not a DataFrame, generic feature names ('feature_1', 'feature_2', etc.)
        will be used.

    Attributes
    ----------
    dataframe_ : pd.DataFrame
        Stores the data passed through the transformer during the `transform` step as a DataFrame.

    Methods
    -------
    fit(X, y=None)
        No-op; returns self.

    transform(X)
        Stores the data and returns it unchanged.
    """

    def __init__(self, feature_names=None):
        self.dataframe_ = None
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # If X is already a DataFrame, store it directly
            self.dataframe_ = X.copy()
        elif isinstance(X, np.ndarray) or issparse(X):
            # Handle NumPy array or sparse matrix
            if self.feature_names is None:
                # Generate generic feature names if none provided
                n_features = X.shape[1]
                self.feature_names = [f"feature_{i+1}" for i in range(n_features)]
            elif len(self.feature_names) != X.shape[1]:
                raise ValueError("Number of provided feature names does not match number of features in the data.")
            
            # Convert to DataFrame with feature names
            self.dataframe_ = pd.DataFrame(X.toarray() if issparse(X) else X, columns=self.feature_names)
        else:
            raise TypeError("Input must be a Pandas DataFrame, NumPy array, or sparse matrix.")

        return X