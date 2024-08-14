import pandas as pd
from sklearn.base import TransformerMixin


class CleanFeatureNames(TransformerMixin):
    """
    Transformer that cleans feature names by replacing specified characters or strings
    that may not be allowed by certain models, such as XGBoost.

    By default, this transformer replaces '[' with '(' and ']' with ')', but users can specify
    other string parts to replace as needed.

    This is particularly useful, for example, when using the DecisionTreeFeatures transformer
    from the feature-engine library, which can produce features with brackets '[' in the resulting
    feature names that are not compatible with XGBoost.

    Parameters
    ----------
    replacements : dict, default=None
        A dictionary where keys are the strings to be replaced and values are the corresponding replacements.
        For example, to replace '[' with '(', use `{'[': '('}`. If None, the default replacements
        {'[': '(', ']': ')'} will be used.

    Methods
    -------
    fit(X, y=None)
        No-op; returns self.

    transform(X)
        Replaces specified characters in the DataFrame column names and
        returns the modified DataFrame.

    Parameters
    ----------
    X : pandas DataFrame
        The input data with feature names to be cleaned.

    Returns
    -------
    X : pandas DataFrame
        The DataFrame with cleaned feature names.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame.

    Examples
    --------
    >>> from feature_engine.creation import DecisionTreeFeatures
    >>> transformer = DecisionTreeFeatures(variables=['feature1', 'feature2'], random_state=42)
    >>> X_transformed = transformer.fit_transform(X)
    >>> cleaner = CleanFeatureNames(replacements={'[': '(', ']': ')'})
    >>> X_cleaned = cleaner.transform(X_transformed)
    """

    def __init__(self, replacements=None):
        if replacements is None:
            # Default replacements for compatibility with models like XGBoost
            replacements = {'[': '(', ']': ')'}
        self.replacements = replacements

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Apply all specified replacements to column names
        for old, new in self.replacements.items():
            X.columns = X.columns.str.replace(old, new, regex=False)

        return X