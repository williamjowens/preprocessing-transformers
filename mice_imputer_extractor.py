import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Union


class MiceImputerExtractor(BaseEstimator, TransformerMixin):
    """
    A transformer to extract a specific DataFrame from the list of tuples returned by `MiceImputer`
    from the `autoimpute` library.

    This transformer allows selection of a DataFrame from the output of the `MiceImputer` based on an
    index, or predefined options such as 'first', 'last', 'mean', or 'random'. The selected DataFrame
    is returned for further processing in a pipeline.

    Parameters
    ----------
    selection : int or str, default='first'
        The selection criterion for extracting a DataFrame.
        - If an integer is provided, it specifies the 1-based index of the DataFrame to select.
        - 'first': Selects the first DataFrame in the list.
        - 'last': Selects the last DataFrame in the list.
        - 'mean': Averages all the DataFrames in the list and returns the mean DataFrame.
        - 'random': Selects a random DataFrame from the list.

    Attributes
    ----------
    selection : int or str
        Stores the selection criterion.

    Examples
    --------
    >>> from autoimpute.imputations import MiceImputer
    >>> imputer = MiceImputer()
    >>> imputed_data = imputer.fit_transform(data)
    >>> extractor = MiceImputerExtractor(selection='mean')
    >>> transformed_data = extractor.fit_transform(imputed_data)
    """

    def __init__(self, selection: Union[int, str] = 'first'):
        self.selection = selection

    def fit(self, X: List[Tuple[int, pd.DataFrame]], y: np.ndarray = None) -> 'MiceImputerExtractor':
        """
        Fit method. This transformer does not learn anything, so it simply returns self.

        Parameters
        ----------
        X : list of tuples
            Input data to fit. Each tuple should consist of an integer and a DataFrame.

        y : array-like, default=None
            Target values (ignored).

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X: List[Tuple[int, pd.DataFrame]]) -> pd.DataFrame:
        """
        Transform the input data by selecting the specified DataFrame from the list of tuples.

        Parameters
        ----------
        X : list of tuples
            The input data, where each element is a tuple consisting of an integer and a DataFrame.

        Returns
        -------
        df : pandas DataFrame
            The selected DataFrame based on the `selection` criterion.

        Raises
        ------
        TypeError
            If the input is not a list of tuples.
        ValueError
            If the selection integer is out of range or if an invalid selection option is provided.
        """
        if not isinstance(X, list) or not all(isinstance(item, tuple) for item in X):
            raise TypeError(f"Input must be a list of tuples. Got {type(X)} instead.")
        if not X:
            raise ValueError("Input list is empty. Cannot extract DataFrame.")

        if isinstance(self.selection, int):
            # Handle integer-based selection (1-based index)
            if self.selection < 1 or self.selection > len(X):
                raise ValueError(f"Selection integer must be between 1 and {len(X)}")
            _, df = X[self.selection - 1]

        elif self.selection == 'first':
            _, df = X[0]
        elif self.selection == 'last':
            _, df = X[-1]
        elif self.selection == 'mean':
            dfs = [df for _, df in X]
            df = sum(dfs) / len(dfs)
        elif self.selection == 'random':
            _, df = random.choice(X)
        else:
            raise ValueError("Invalid selection. Choose from an integer, 'first', 'last', 'mean', or 'random'.")

        return df