from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np
import pandas as pd


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    A transformer that wraps around an outlier detection estimator to remove or replace outliers.

    This transformer can be used to preprocess data by identifying outliers using a scikit-learn
    compatible outlier detection model. It can then either replace the detected outliers with NaN
    values or remove them entirely. The output format of the transformed data can be customized.

    Parameters
    ----------
    outlier_detector : scikit-learn compatible estimator
        An outlier detector that implements `.fit()` and `.predict()` methods.

    action : str, default='replace'
        Action to take on detected outliers.
        'replace' will replace outliers with NaN values.
        'remove' will remove the outliers entirely.

    output_type : str, default='same'
        Specifies the format of the output data.
        'same' will return the data in the same format as the input.
        'pandas' will return the data as a pandas DataFrame.
        'numpy' will return the data as a numpy array.

    Raises
    ------
    ValueError
        If an invalid `action` or `output_type` is provided.
    """

    def __init__(self, outlier_detector, action='replace', output_type='same'):
        self.outlier_detector = outlier_detector
        self.action = action
        self.output_type = output_type
        self._validate_params()

    def _validate_params(self):
        """Validate the parameters of the transformer."""
        if self.action not in ['replace', 'remove']:
            raise ValueError("Invalid action specified. Choose 'replace' or 'remove'.")
        if self.output_type not in ['same', 'pandas', 'numpy']:
            raise ValueError("Invalid output_type specified. Choose 'same', 'pandas', or 'numpy'.")

    def fit(self, X, y=None, **fit_params):
        """
        Fits the underlying outlier detector to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), default=None
            Target values.

        **fit_params : dict
            Additional parameters to pass to the outlier detector's fit method.

        Returns
        -------
        self : OutlierRemover
            The fitted transformer.
        """
        self.outlier_detector.fit(X, y, **fit_params)
        return self

    def transform(self, X, y=None):
        """
        Transforms the data by either replacing or removing outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        X_transformed : array-like
            The transformed data with outliers handled as specified.
        """
        check_is_fitted(self, 'outlier_detector')
        X_orig_type = type(X)

        # Store the original index and columns if applicable
        original_index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else None
        original_columns = X.columns if isinstance(X, pd.DataFrame) else None

        # Convert X to numpy array for internal processing
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_values = X.values
        else:
            X_values = check_array(X, ensure_2d=True)

        # Predict outliers
        outliers = self.outlier_detector.predict(X_values) == -1

        if self.action == 'replace':
            # Replace outliers with NaN
            X_values[outliers] = np.nan
        elif self.action == 'remove':
            # Remove outliers
            X_values = X_values[~outliers]

        # Convert back to the desired output type
        if self.output_type == 'same':
            if X_orig_type == pd.DataFrame:
                X_transformed = pd.DataFrame(X_values, index=original_index, columns=original_columns)
            elif X_orig_type == pd.Series:
                X_transformed = pd.Series(X_values, index=original_index, name=X.name)
            else:
                X_transformed = X_values
        elif self.output_type == 'pandas':
            X_transformed = pd.DataFrame(X_values, index=original_index)
        elif self.output_type == 'numpy':
            X_transformed = np.array(X_values)
        else:
            raise ValueError("Invalid output_type specified. Choose 'same', 'pandas', or 'numpy'.")

        return X_transformed

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits the outlier detector and transforms the data in a single step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit and transform.

        y : array-like of shape (n_samples,), default=None
            Target values.

        **fit_params : dict
            Additional fitting parameters to pass to the outlier detector's fit method.

        Returns
        -------
        X_transformed : array-like
            The transformed data with outliers handled as specified.
        """
        return self.fit(X, y, **fit_params).transform(X, y)