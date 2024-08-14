import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.base import OneToOneFeatureMixin
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


class GelmanScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    A scikit-learn compatible transformer that applies Gelman scaling to numerical features.

    Gelman scaling standardizes each feature by centering it to have a mean of zero and
    scaling it by dividing by two times its standard deviation. This scaling technique
    is particularly useful in hierarchical or multilevel models to ensure coefficients
    are on a comparable scale and more interpretable.

    Attributes
    ----------
    means_ : ndarray of shape (n_features,)
        The mean of each feature in the training set.

    scales_ : ndarray of shape (n_features,)
        The scale (2 * standard deviation) of each feature in the training set.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    def __init__(self):
        self.means_ = None
        self.scales_ = None

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary."""
        if hasattr(self, "means_"):
            del self.means_
            del self.scales_

    def fit(self, X, y=None):
        """
        Compute the mean and scale (2 * standard deviation) for each feature in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature mean and scale.

        y : None
            Ignored, exists for compatibility with sklearn pipeline.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # Reset internal state before fitting
        self._reset()

        # Validate the input data
        X = self._validate_data(X, dtype=np.float64, force_all_finite="allow-nan")

        # Calculate means and scales
        self.means_ = np.nanmean(X, axis=0)
        self.scales_ = 2 * np.nanstd(X, axis=0)

        # Save the number of features seen during fit
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """
        Apply Gelman scaling to the input data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_scaled : array-like of shape (n_samples, n_features)
            The transformed data with Gelman scaling applied.
        """
        check_is_fitted(self, ["means_", "scales_"])
        X = self._validate_data(X, reset=False, dtype=np.float64, force_all_finite="allow-nan")

        # Apply Gelman scaling
        return (X - self.means_) / self.scales_

    def inverse_transform(self, X):
        """
        Reverse the Gelman scaling on the input data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to inverse transform.

        Returns
        -------
        X_unscaled : array-like of shape (n_samples, n_features)
            The data after reversing the Gelman scaling.
        """
        check_is_fitted(self, ["means_", "scales_"])
        X = self._validate_data(X, reset=False, dtype=np.float64, force_all_finite="allow-nan")

        # Reverse Gelman scaling
        return (X * self.scales_) + self.means_

    def _more_tags(self):
        return {"allow_nan": True}