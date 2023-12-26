"""Calculation of fairness sequentially with respect to all orders of sensitive variables"""

import numpy as np

from ....metrics._fairness_metrics import unfairness_dict


def unfairness_permutations(permut_y_fair_dict: dict[tuple, dict[str, np.ndarray]], all_combs_sensitive_features: dict[tuple, np.ndarray]) -> list[dict[str, float]]:
    """
    Compute unfairness values for multiple fair output datasets and multiple sensitive attribute datasets.

    Parameters
    ----------
    permut_y_fair_dict : dict
        A dictionary containing permutations of fair output datasets.
    all_combs_sensitive_features : dict
        A dictionary containing combinations of columns permutations for sensitive attribute datasets.

    Returns
    -------
    list
        A list of dictionaries containing unfairness values for each permutation of fair output datasets.

    Example
    -------
    >>> permut_y_fair_dict = {(1,2): {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])},
    ...                        (2,1): {'Base model':np.array([19,39,65]), 'sens_var_2':np.array([34,39,60]), 'sens_var_1':np.array([28,39,42])}}
    >>> all_combs_sensitive_features = {(1,2): np.array([['blue', 2], ['red', 9], ['green', 5]]),
    ...                                (2,1): np.array([[2, 'blue'], [9, 'red'], [5, 'green']])}
    >>> unfs_list = compute_unfairness_permutations(permut_y_fair_dict, all_combs_sensitive_features)
    >>> print(unfs_list)
    [{'Base model': 46.0, 'sens_var_1': 28.0, 'sens_var_2': 14.0}, 
     {'Base modem': 46.0, 'sens_var_1': 26.0, 'sens_var_2': 14.0}]
    """
    unfs_list = []
    for key, value in permut_y_fair_dict.items():
        unfs_list.append(unfairness_dict(
            value, np.array(all_combs_sensitive_features[key])))
    return unfs_list
