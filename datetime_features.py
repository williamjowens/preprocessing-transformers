from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import holidays


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer that extracts various datetime features from a specified datetime column or index in the data.
    This transformer can handle Pandas DataFrames, Series, and NumPy arrays, and can output the transformed data
    in the desired format.

    Parameters
    ----------
    output_format : str, default='dataframe'
        Format of the output after transformation. Can be 'dataframe' to return a Pandas DataFrame
        or 'numpy' to return a NumPy array.

    Attributes
    ----------
    datetime_column : str or None
        The name of the datetime column identified during fitting. If None, the datetime information is assumed
        to be in the index.

    granularity : str or None
        The granularity of the datetime information, which can be 'hour', 'day', 'week', 'month', 'quarter', or 'year'.
        This determines which features will be extracted during transformation.

    Methods
    -------
    fit(X, y=None)
        Identifies the datetime column or index in the data and determines the granularity of the datetime information.

    transform(X)
        Extracts the relevant datetime features based on the identified granularity and returns the transformed data.

    fit_transform(X, y=None)
        Fits the transformer to X and returns the transformed data in a single step.

    _convert_to_dataframe(X)
        Converts the input X into a Pandas DataFrame, if it is not already one.

    _convert_output_format(X)
        Converts the transformed DataFrame into the specified output format (DataFrame or NumPy array).
    """

    def __init__(self, output_format='dataframe'):
        self.datetime_column = None
        self.granularity = None
        self.output_format = output_format
        self.holidays = holidays.US()

    def fit(self, X, y=None):
        """
        Identifies the datetime column or index in the data and determines the granularity of the datetime information.
        The granularity could be 'hour', 'day', 'week', 'month', 'quarter', or 'year', depending on the available datetime data.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            The input data to fit the transformer. This can be a Pandas DataFrame, Series, or NumPy array.

        y : Ignored
            Not used, present here for consistency with the scikit-learn API.

        Returns
        -------
        self : DateTimeFeatures
            The fitted instance of the transformer.
        """
        X = self._convert_to_dataframe(X)

        datetime_columns = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            self.datetime_column = datetime_columns[0]
            dt_series = pd.to_datetime(X[self.datetime_column])
        else:
            dt_series = X.index.to_series()

        if dt_series.dt.hour.notnull().all():
            self.granularity = 'hour'
        elif dt_series.dt.day.notnull().all():
            self.granularity = 'day'
        elif dt_series.dt.isocalendar().week.notnull().all():
            self.granularity = 'week'
        elif dt_series.dt.month.notnull().all():
            self.granularity = 'month'
        elif dt_series.dt.quarter.notnull().all():
            self.granularity = 'quarter'
        elif dt_series.dt.year.notnull().all():
            self.granularity = 'year'
        else:
            raise ValueError("Could not determine the granularity of the datetime information.")

        return self

    def transform(self, X):
        """
        Extracts relevant datetime features from the data based on the identified granularity.
        Depending on the granularity, the features extracted may include year, quarter, month, week,
        day, hour, minute, second, day of the week, day of the month, day of the year, and whether the date is a holiday.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            The input data to transform. This can be a Pandas DataFrame, Series, or NumPy array.

        Returns
        -------
        X_transformed : DataFrame or ndarray
            The transformed data with extracted datetime features.
            The output format is determined by the 'output_format' parameter.
        """
        X_copy = self._convert_to_dataframe(X)

        if self.datetime_column:
            dt_series = pd.to_datetime(X_copy[self.datetime_column])
            X_copy = X_copy.drop(columns=[self.datetime_column])
        else:
            dt_series = X_copy.index.to_series()

        X_copy['year'] = dt_series.dt.year
        if self.granularity in ['quarter', 'month', 'week', 'day', 'hour']:
            X_copy['quarter'] = dt_series.dt.quarter
        if self.granularity in ['month', 'week', 'day', 'hour']:
            X_copy['month'] = dt_series.dt.month
        if self.granularity in ['week', 'day', 'hour']:
            X_copy['week'] = dt_series.dt.isocalendar().week
        if self.granularity in ['day', 'hour']:
            X_copy['day'] = dt_series.dt.day
            X_copy['dayofmonth'] = dt_series.dt.day
            X_copy['dayofyear'] = dt_series.dt.dayofyear
            X_copy['weekend'] = dt_series.dt.dayofweek >= 5
            X_copy['holiday'] = dt_series.dt.date.isin(self.holidays)
        if self.granularity == 'hour':
            X_copy['hour'] = dt_series.dt.hour
            X_copy['minute'] = dt_series.dt.minute
            X_copy['second'] = dt_series.dt.second
        if self.granularity in ['week', 'day', 'hour']:
            X_copy['dayofweek'] = dt_series.dt.dayofweek

        return self._convert_output_format(X_copy)

    def fit_transform(self, X, y=None):
        """
        Fits the transformer to X and transforms the data in a single step.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            The input data to fit and transform. This can be a Pandas DataFrame, Series, or NumPy array.

        y : Ignored
            Not used, present here for consistency with the scikit-learn API.

        Returns
        -------
        X_transformed : DataFrame or ndarray
            The transformed data with extracted datetime features.
            The output format is determined by the 'output_format' parameter.
        """
        return self.fit(X, y).transform(X)

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