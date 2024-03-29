"""Make predictions fair sequentially with respect to all orders of sensitive variables"""

import itertools
import numpy as np

from ...fairness._wasserstein import MultiWasserstein
from typing import Optional


def permutations_columns(sensitive_features: np.ndarray) -> dict[tuple, list]:
    """
    Generate permutations of columns in the input array sensitive_features.

    Parameters
    ----------
    sensitive_features : np.ndarray, shape (n_samples, n_sensitive_features)
        Input array where each column represents a different sensitive feature.

    Returns
    -------
    dict
        A dictionary where keys are tuples representing permutations of column indices,
        and values are corresponding permuted arrays of sensitive features.

    Example
    -------
    >>> sensitive_features = [[1, 2], [3, 4], [5, 6]]
    >>> generate_permutations_cols(sensitive_features)
    {(1, 2): [[1, 2], [3, 4], [5, 6]], (2, 1): [[3, 4], [1, 2], [5, 6]]}

    Note
    ----
    This function generates all possible permutations of columns and stores them in a dictionary.
    """
    n = len(sensitive_features[0])
    ind_cols = list(range(n))
    permut_cols = list(itertools.permutations(ind_cols))
    sensitive_features_with_ind = np.vstack((ind_cols, sensitive_features))

    dict_all_combs = {}
    for permutation in permut_cols:
        permuted_sensitive_features = sensitive_features_with_ind[:, permutation]

        key = tuple(permuted_sensitive_features[0]+1)

        values = permuted_sensitive_features[1:].tolist()
        dict_all_combs[key] = values

    return dict_all_combs


def calculate_perm_wasserstein(y_calib: np.ndarray, sensitive_features_calib: np.ndarray, y_test: np.ndarray, sensitive_features_test: np.ndarray, epsilon: Optional[list[float]] = None):
    """
    Calculate Wasserstein distance for different permutations of sensitive features between calibration and test sets.

    Parameters
    ----------
    y_calib : np.ndarray, shape (n_samples,)
        Calibration set predictions.
    sensitive_features_calib : np.ndarray, shape (n_samples, n_sensitive_features)
        Calibration set sensitive features.
    y_test : np.ndarray, shape (n_samples,)
        Test set predictions.
    sensitive_features_test : np.ndarray, shape (n_samples, n_sensitive_features)
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
            np.array([np.array(epsilon).T]))
        for key in all_perm_epsilon.keys():
            all_perm_epsilon[key] = all_perm_epsilon[key][0]

    store_dict = {}
    for key in all_perm_calib:
        wst = MultiWasserstein()
        wst.fit(y_calib, np.array(all_perm_calib[key]))
        if epsilon is None:
            wst.transform(y_test, np.array(
                all_perm_test[key]))
        else:
            wst.transform(y_test, np.array(
                all_perm_test[key]), all_perm_epsilon[key])
        store_dict[key] = wst.y_fair
        old_keys = list(store_dict[key].keys())
        new_keys = ['Base model'] + [f'sens_var_{k}' for k in key]
        key_mapping = dict(zip(old_keys, new_keys))
        store_dict[key] = {key_mapping[old_key]                           : value for old_key, value in store_dict[key].items()}
    return store_dict
