"""Make predictions fair sequentially with respect to all orders of sensitive variables"""

import itertools
import numpy as np
import pandas as pd

from ...fairness._wasserstein import MultiWasserstein
from typing import Optional


def permutations_columns(sensitive_features: pd.DataFrame) -> dict[tuple, list]:
    """
    Generate permutations of columns in the input array sensitive_features.

    Parameters
    ----------
    sensitive_features : pd.DataFrame, shape (n_samples, n_sensitive_features)
        Input array where each column represents a different sensitive feature.

    Returns
    -------
    dict
        A dictionary where keys are tuples representing permutations of column indices,
        and values are corresponding permuted pandas dataframes of sensitive features.

    Example
    -------
    >>> sensitive_features = [[1, 2], [3, 4], [5, 6]]
    >>> generate_permutations_cols(sensitive_features)
    {(1, 2): [[1, 2], [3, 4], [5, 6]], (2, 1): [[3, 4], [1, 2], [5, 6]]}

    Note
    ----
    This function generates all possible permutations of columns and stores them in a dictionary.
    """
    n = sensitive_features.shape[1]
    cols = list(sensitive_features.columns)
    permut_cols = list(itertools.permutations(cols))

    dict_all_combs = {}
    for permutation in permut_cols:
        permuted_sensitive_features = sensitive_features[list(permutation)]
        dict_all_combs[permutation] = permuted_sensitive_features
    return dict_all_combs


def calculate_perm_wasserstein(y_calib: np.ndarray, sensitive_features_calib: pd.DataFrame, y_test: np.ndarray, sensitive_features_test: pd.DataFrame, epsilon: Optional[list[float]] = None):
    """
    Calculate Wasserstein distance for different permutations of sensitive features between calibration and test sets.

    Parameters
    ----------
    y_calib : np.ndarray, shape (n_samples,)
        Calibration set predictions.
    sensitive_features_calib : pd.DataFrame, shape (n_samples, n_sensitive_features)
        Calibration set sensitive features.
    y_test : np.ndarray, shape (n_samples,)
        Test set predictions.
    sensitive_features_test : pd.DataFrame, shape (n_samples, n_sensitive_features)
        Test set sensitive features.
    epsilon : np.ndarray, shape (n_sensitive_features,) or None, default= None
        Fairness constraints.

    Returns
    -------
    dict
        A dictionary where keys are tuples representing permutations of column indices,
        and values are corresponding sequential fairness values for each permutation.

    Example
    -------
    >>> y_calib = [1, 2, 3]
    >>> sensitive_features_calib = [[1, 2], [3, 4], [5, 6]]
    >>> y_test = [4, 5, 6]
    >>> sensitive_features_test = [[7, 8], [9, 10], [11, 12]]
    >>> calculate_perm_wst(y_calib, sensitive_features_calib, y_test, sensitive_features_test)
    {(1,2): {'Base model': 0.5, 'sens_var_1': 0.2, 'sens_var_2': 0}, (2, 1): {'Base model': 0.3, 'sens_var_2': 0.6, 'sens_var_1': 0.6}}

    Note
    ----
    This function calculates Wasserstein distance for different permutations of sensitive features
    between calibration and test sets and stores the sequential fairness values in a dictionary.
    """
    all_perm_calib = permutations_columns(sensitive_features_calib)
    all_perm_test = permutations_columns(sensitive_features_test)
    if epsilon is not None:
        all_perm_epsilon = permutations_columns(
            pd.DataFrame([epsilon], columns=sensitive_features_calib.columns))

    store_dict = {}
    for key in all_perm_calib:
        wst = MultiWasserstein()
        wst.fit(y_calib, all_perm_calib[key])
        if epsilon is None:
            wst.transform(y_test, all_perm_test[key])
        else:
            wst.transform(y_test, all_perm_test[key], 
                          all_perm_epsilon[key].iloc[0].tolist())
        store_dict[key] = wst.get_sequential_fairness()
    return store_dict
