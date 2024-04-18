"""
Arrow plot of the fairness-performance relationship.
"""

# Authors: Agathe F, Suzie G, Francois H, Philipp R, Arthur C
# License: BSD 3 clause
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import re
import pandas as pd
from typing import Optional, Callable, Union

from ..utils.permutations._compute_permutations import permutations_columns, calculate_perm_wasserstein
from ..utils.permutations.metrics._fairness_permutations import unfairness_permutations
from ..utils.permutations.metrics._performance_permutations import performance_permutations
from ..fairness._wasserstein import MultiWasserstein
from ..metrics._fairness_metrics import unfairness_dict
from ..metrics._performance_metrics import performance_dict


def fair_customized_arrow_plot(unfs_dict: dict[str, np.ndarray],
                               performance_dict: dict[str, np.ndarray],
                               permutations: bool = False,
                               base_model: bool = True,
                               final_model: bool = True) -> plt.Axes:
    """
    Generates an arrow plot representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness.

    Parameters
    ----------
    unfs_dict : dict
        A dictionary containing unfairness values associated with the sequentially fair output datasets.
    performance_dict : dict
        A dictionary containing performance values associated with the sequentially fair output datasets.
    permutations : bool, optional
        If True, displays permutations of arrows based on input dictionaries. Defaults to False.
    base_model : bool, optional
        If True, includes the base model arrow. Defaults to True.
    final_model : bool, optional
        If True, includes the final model arrow. Defaults to True.

    Returns
    -------
    matplotlib.axes.Axes
        arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness.

    Note
    ----
    - This function uses a global variable `ax` for plotting, ensuring compatibility with external code.
    """

    x = []
    y = []
    sens = []

    for i, key in enumerate(unfs_dict.keys()):
        sens.append(key)
        x.append(unfs_dict[key])
        y.append(performance_dict[key])
    
    if len(sens) > 2:
        first_sens = sens[1]
        double_sorted_sens = sorted(sens[1:3])
    else:
        first_label_not_used = True
        double_label_not_used = True
    
    if first_sens not in first_current_sens:
        first_current_sens.append(first_sens)
        first_label_not_used = True
    else:
        first_label_not_used = False
    
    if double_sorted_sens not in double_current_sens:
        double_current_sens.append(double_sorted_sens)
        double_label_not_used = True
    else:
        double_label_not_used = False
    
    global ax

    if not permutations:
        fig, ax = plt.subplots()

    line = ax.plot(x, y, linestyle="--", alpha=0.25, color="grey")[0]

    for i in range(len(sens)):
        if i > 0:
            ax.arrow((x[i-1]+x[i])/2, (y[i-1]+y[i])/2, (x[i]-x[i-1])/10,
                      (y[i]-y[i-1])/10, width = (np.max(y)-np.min(y))/70, 
                      color ="grey")
        if (i == 0) & (base_model):
            line.axes.annotate(f"Base\nmodel", xytext=(
                x[0]-5*np.min(x)/20, y[0]-y[0]/80), xy=(x[0], y[0]), size=10)
            ax.scatter(x[0], y[0], label="Base model", marker="^", 
                       color="darkgrey", s=100)
        elif (i == 1) & (first_label_not_used):
            label = f"{sens[i]}-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        elif (i == len(x)-1) & (final_model):
            label = f"Final fair model"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="*", s=150,
                       color="#d62728")
        elif (i == 2) & (i < len(x)-1) & (double_label_not_used):
            label = f"{sens[1]}-{sens[i]}-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        elif (i!=0) & (i!=len(x)-1):
            ax.scatter(x[i], y[i], marker="+", s=150, color="grey", alpha=0.4)
    ax.set_xlabel("Unfairness")
    ax.set_ylabel("Performance")
    ax.set_title("Exact fairness")
    ax.legend(loc="lower right")
    #ax.autoscale_view()
    return ax

def fair_arrow_plot(sensitive_features_calib: pd.DataFrame,
                    sensitive_features_test: pd.DataFrame,
                    y_calib: np.ndarray,
                    y_test: np.ndarray,
                    y_true_test: np.ndarray,
                    epsilon: Optional[float] = None,
                    metric: Callable = mean_squared_error,
                    threshold: Optional[float] = None,
                    positive_class: Union[int, str] = 1) -> plt.Axes:
    """
    Generates an arrow plot representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness.

    Parameters
    ----------
    sensitive_features_calib : pd.DataFrame
        Sensitive features for calibration.
    sensitive_features_test : pd.DataFrame
        Sensitive features for testing.
    y_calib : numpy.ndarray
        Predictions for calibration.
    y_test : numpy.ndarray
        Predictions for testing.
    y_true_test : numpy.ndarray
        True labels for testing.
    epsilon : float, optional, default = None
        Epsilon value for calculating Wasserstein distance
    metric : Callable, default = sklearn.mean_squared_error
        The metric used to evaluate performance.
    threshold : float, default = None
        The threshold used to transform scores from binary classification into labels for evaluation of performance.
    positive_class : int or str, optional, default=1
        The positive class label used for applying threshold in the case of binary classification. Can be either an integer or a string.

    Returns
    -------
    matplotlib.axes.Axes
        arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness.

    Note
    ----
    This function uses a global variable `ax` for plotting, ensuring compatibility with external code.

    Examples
    --------
    >>> from sklearn.metrics import f1_score
    >>> sensitive_features_calib = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue'], 'nb_child': [1, 2, 0, 2]})
    >>> sensitive_features_test = pd.DataFrame({'color': ['blue', 'blue', 'blue', 'green'], 'nb_child': [3, 2, 1, 2]})
    >>> y_calib = np.array([0.6, 0.43, 0.32, 0.8])
    >>> y_test = np.array([0.8, 0.35, 0.23, 0.2])
    >>> y_true_test = np.array(['no', 'no', 'yes', 'no'])
    >>> fair_arrow_plot(sensitive_features_calib, sensitive_features_test, y_calib, y_test, y_true_test, f1_score, threshold=0.5, positive_class='yes')
    
    """
    global ax
    global double_current_sens
    double_current_sens = []
    global first_current_sens
    first_current_sens = []

    exact_wst = MultiWasserstein()
    exact_wst.fit(y_calib, sensitive_features_calib)
    y_final_fair = exact_wst.transform(y_test, sensitive_features_test, epsilon=epsilon)
    y_sequential_fair = exact_wst.get_sequential_fairness()

    unfs_dict = unfairness_dict(y_sequential_fair, sensitive_features_test)
    perf_dict = performance_dict(y_true_test, y_sequential_fair, metric=metric, threshold=threshold, 
                                 positive_class=positive_class)
    
    return fair_customized_arrow_plot(unfs_dict=unfs_dict, performance_dict=perf_dict)

def _fair_customized_multiple_arrow_plot(unfs_list: list[dict[str, np.ndarray]],
                                         performance_list: list[dict[str, np.ndarray]]) -> plt.Axes:
    """
    Plot arrows representing the fairness-performance ccombinations step by step (by sensitive attribute) to reach fairness for all permutations
    (order of sensitive variables for which fairness is calculated).

    Parameters
    ----------
    unfs_list : list
        A list of dictionaries containing unfairness values for each permutation of fair output datasets.
    performance_list : list
        A list of dictionaries containing performance values for each permutation of fair output datasets.

    Returns
    -------
    matplotlib.axes.Axes
        arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness for each combination.

    Note
    ----
    This function uses a global variable `ax` for plotting, ensuring compatibility with external code.
    """
    global ax
    global double_current_sens
    double_current_sens = []
    global first_current_sens
    first_current_sens = []
    fig, ax = plt.subplots()
    for i in range(len(unfs_list)):
        if i == 0:
            fair_customized_arrow_plot(unfs_list[i], performance_list[i],
                                       permutations=True, final_model=False)
        elif i == len(unfs_list)-1:
            fair_customized_arrow_plot(unfs_list[i], performance_list[i], 
                                       permutations=True, base_model=False)
        else:
            fair_customized_arrow_plot(unfs_list[i], performance_list[i], 
                                       permutations=True, base_model=False, final_model=False)
    return ax


def fair_multiple_arrow_plot(sensitive_features_calib: pd.DataFrame,
                             sensitive_features_test: pd.DataFrame,
                             y_calib: np.ndarray,
                             y_test: np.ndarray,
                             y_true_test: np.ndarray,
                             epsilon: Optional[float] = None,
                             metric: Callable = mean_squared_error,
                             threshold: Optional[float] = None,
                             positive_class: Union[int, str] = 1) -> plt.Axes:
    """
    Plot arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness for different permutations.

    Parameters
    ----------
    sensitive_features_calib : pd.DataFrame
        Sensitive features for calibration.
    sensitive_features_test : pd.DataFrame
        Sensitive features for testing.
    y_calib : numpy.ndarray
        Predictions for calibration.
    y_test : numpy.ndarray
        Predictions for testing.
    y_true_test : numpy.ndarray
        True labels for testing.
    epsilon : float, optional, default = None
        Epsilon value for calculating Wasserstein distance
    metric : Callable, default = sklearn.mean_squared_error
        The metric used to evaluate performance.
    threshold : float, default = None
        The threshold used to transform scores from binary classification into labels for evaluation of performance.
    positive_class : int or str, optional, default=1
        The positive class label used for applying threshold in the case of binary classification. Can be either an integer or a string.

    Returns
    -------
    matplotlib.axes.Axes
        Arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness for different permutations.
    
    Note
    ----
    This function uses a global variable `ax` for plotting, ensuring compatibility with external code.

    Examples
    --------
    >>> from sklearn.metrics import f1_score
    >>> sensitive_features_calib = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue'], 'nb_child': [1, 2, 0, 2]})
    >>> sensitive_features_test = pd.DataFrame({'color': ['blue', 'blue', 'blue', 'green'], 'nb_child': [3, 2, 1, 2]})
    >>> y_calib = np.array([0.6, 0.43, 0.32, 0.8])
    >>> y_test = np.array([0.8, 0.35, 0.23, 0.2])
    >>> y_true_test = np.array(['no', 'no', 'yes', 'no'])
    >>> fair_multiple_arrow_plot(sensitive_features_calib, sensitive_features_test, y_calib, y_test, y_true_test, f1_score, threshold=0.5, positive_class='yes')

    """
    permut_y_fair_dict = calculate_perm_wasserstein(
        y_calib, sensitive_features_calib, y_test, sensitive_features_test, epsilon=epsilon)
    all_combs_sensitive_features_test = permutations_columns(
        sensitive_features_test)
    unfs_list = unfairness_permutations(
        permut_y_fair_dict, all_combs_sensitive_features_test)
    performance_list = performance_permutations(
        y_true_test, permut_y_fair_dict, metric=metric, threshold=threshold, positive_class=positive_class)
    return _fair_customized_multiple_arrow_plot(unfs_list, performance_list)