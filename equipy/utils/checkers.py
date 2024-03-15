"""Check inputs formats."""

import numpy as np
import pandas as pd
import warnings
from typing import Callable, Dict, Any, Optional, Union
from sklearn.metrics import mean_squared_error

def get_subtype(arr):
    """
    Returns the subtype of the elements in the NumPy array.

    Parameters
    ----------
    arr : np.ndarray

    Raises
    ------
    Subtype of the elements in the array (e.g., 'np.integer', 'np.floating', etc.)
    """
    if np.issubdtype(arr.dtype, np.integer):
        return np.integer
    elif np.issubdtype(arr.dtype, np.floating):
        return np.floating
    elif np.issubdtype(arr.dtype, np.bool_):
        return np.bool_
    elif np.issubdtype(arr.dtype, np.character):
        return np.character
    else:
        return arr.dtype
    
def _check_type(y_true: np.ndarray, y_dict: Dict[Any, Dict[Any, Any]], threshold: Optional[float] = None) -> None:
    """
    Check the types of observed events and fair outputs.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Observed, true values.
    y_dict : dict
        Dictionary containing fair predictions for different permutations of sensitive features. Each value is itself a dictionary
        representing fair predictions for a specific permutation of sensitive features.
    threshold : float, default = None
        The threshold used to transform scores from binary classification into labels for evaluation of performance.

    Raises
    ------
    ValueError
        If y_true and the values in y_dict are not of the same type.
    """
    type_y_true = get_subtype(y_true)
    type_y_fair = get_subtype(list(y_dict.values())[0])

    if type_y_true != type_y_fair and threshold is None:
        raise ValueError(
            "Specify a threshold to transform scores into labels when using a classification performance metric")
    
def _check_positive_class(y_true: np.ndarray, positive_class: Union[int, str] = 1) -> None:
    """
    Check the types of observed events and fair outputs.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Observed, true values.
   positive_class : int or str, optional, default=1
        The positive class label used for applying threshold in the case of binary classification. Can be either an integer or a string.

    Raises
    ------
    ValueError
        If y_true and the values in y_dict are not of the same type.
    """
    y_true_modalities = list(set(y_true))

    if (positive_class == 1) and (1 not in y_true_modalities):
        raise ValueError(
            "Specify a positive class if using positive labels other than 1")


def _check_metric(y: np.ndarray, metric: Callable) -> None:
    """
    Check that it is regression and not classification.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Observed, true values.
    metric : Callable,
        The metric used to compute the performance.
    Raises
    ------
    Warning 
        If it is classification.
    """
    if np.all(np.isin(y, [0, 1])) and (metric ==  mean_squared_error):
        warnings.warn(
            "You used mean squared error as metric but it looks like you are using classification scores")


def _check_nb_observations(sensitive_features: pd.DataFrame) -> None:
    """
    Check that there is more than one observation.

    Parameters
    ----------
    sensitive_features : pd.DataFrame, shape (n_samples, n_sensitive_features)
        The calibration samples representing multiple sensitive attributes.

    Raises
    ------
    ValueError
        If there is only a single observation
    """
    if len(sensitive_features) == 1:
        raise ValueError("Fairness correction can not be applied on a single observation")


def _check_shape(y: np.ndarray, sensitive_features: pd.DataFrame) -> None:
    """
    Check the shape and data types of input arrays y and sensitive_feature.

    Parameters
    ----------
    y :  np.ndarray, shape (n_samples,)
        Target values of the data.
    sensitive_features : pd.DataFrame, shape (n_samples, n_sensitive_features)
        Input samples representing the sensitive attributes.

    Raises
    ------
    ValueError
        If the input arrays have incorrect shapes or data types.
    """
    if not isinstance(sensitive_features, pd.DataFrame):
        raise ValueError('sensitive_features must be a pandas DataFrame')

    if not isinstance(y, np.ndarray):
        raise ValueError('y must be an array')

    if len(sensitive_features) != len(y):
        raise ValueError(
            'sensitive_features and y should have the same length')
    
    if not (np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.integer)):
        raise ValueError('y should contain only float or integer numbers')

def _check_unique_mod(sensitive_feature: pd.DataFrame) -> None:
    """
    Check the shape and data types of input arrays y and sensitive_feature.

    Parameters
    ----------
    sensitive_feature : pd.DataFrame, shape (n_samples, 1)
        Input samples representing the sensitive attribute.

    Raises
    ------
    ValueError
        If the input of sensitive feature contains a unique modality.
    """

    if len(np.unique(sensitive_feature)) == 1:
        raise ValueError(
            "At least one of your sensitive attributes contains only one modality and so it is already fair. Remove it from your pandas DataFrame of sensitive features.")
    

def _check_col(columns_calib: pd.core.indexes.base.Index, columns_test: pd.core.indexes.base.Index) -> None:
    """
    Check if columns of sensitive_features_calib and sensitive_features_test are similar.

    Parameters
    ----------
    sensitive_features_calib : pd.DataFrame
        DataFrame of sensitive attributes of the calibration data.
    sensitive_features_test : pd.DataFrame
        DataFrame of sensitive attributes of the test data.

    Raises
    ------
    ValueError
        If names of columns or number of columns are different between calibration and test data.
    """

    if len(columns_calib) != len(columns_test):
        raise ValueError(
            "Your calibration sensitive features should contain the same number of columns than your test sensitive features")
    elif set(columns_calib) != set(columns_test):
        raise ValueError(
            "Your calibration sensitive features should contain the same column names than your test sensitive features")
    elif list(columns_calib) != list(columns_test):
        raise ValueError(
            "The columns in your calibration sensitive features should be in the same order than in your test sensitive features")

def _check_mod(modalities_calib: list, modalities_test: list) -> None:
    """
    Check if modalities in test data are included in calibration data's modalities.

    Parameters
    ----------
    modalities_calib : list
        Modalities from the calibration data.
    modalities_test : list
        Modalities from the test data.

    Raises
    ------
    ValueError
        If modalities in test data are not present in calibration data.
    """
    missing_modalities = set(modalities_test) - set(modalities_calib)
    if len(missing_modalities) != 0:
        raise ValueError(
            f"The following modalities of the test sensitive features are not in modalities of the calibration sensitive features: {missing_modalities}")


def _check_epsilon(epsilon: float) -> None:
    """
    Check if epsilon (fairness parameter) is within the valid range [0, 1].

    Parameters
    ----------
    epsilon : float
        Fairness parameter controlling the trade-off between fairness and accuracy.

    Raises
    ------
    ValueError
        If epsilon is outside the valid range [0, 1].
    """
    if epsilon < 0 or epsilon > 1:
        raise ValueError(
            'epsilon must be between 0 and 1')


def _check_epsilon_size(epsilon: list[float], sensitive_features: pd.DataFrame) -> None:
    """
    Check if the epsilon list matches the number of sensitive features.

    Parameters
    ----------
    epsilon : list, shape (n_sensitive_features,)
        Fairness parameters controlling the trade-off between fairness and accuracy for each sensitive feature.

    sensitive_features : pd.DataFrame, shape (n_samples, n_sensitive_features)
        Test samples representing multiple sensitive attributes.

    Raises
    ------
    ValueError
        If the length of epsilon does not match the number of sensitive features.
    """

    if len(epsilon) != sensitive_features.shape[1]:
        raise ValueError(
            'epsilon must have the same length than the number of sensitive features')
