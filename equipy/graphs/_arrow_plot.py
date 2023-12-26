"""Arrow plot of the fairness-performance relationship."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import re
from typing import Optional, Callable

from ..utils.permutations._compute_permutations import permutations_columns, calculate_perm_wasserstein
from ..utils.permutations.metrics._fairness_permutations import unfairness_permutations
from ..utils.permutations.metrics._performance_permutations import performance_permutations


def fair_arrow_plot(unfs_dict: dict[str, np.ndarray],
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
    sens = [0]

    for i, key in enumerate(unfs_dict.keys()):
        x.append(unfs_dict[key])
        if i != 0:
            sens.append(int(''.join(re.findall(r'\d+', key))))
    
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
    
    for key in performance_dict.keys():
        y.append(performance_dict[key])

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
                x[0]+np.min(x)/20, y[0]), xy=(x[0], y[0]), size=10)
            ax.scatter(x[0], y[0], label="Base model", marker="^", s=100)
        elif (i == 1) & (first_label_not_used):
            label = f"$A_{sens[i]}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        elif (i == len(x)-1) & (final_model):
            label = f"$A_{1}$" + r"$_:$" + f"$_{i}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="*", s=150)
        elif (i == 2) & (i < len(x)-1) & (double_label_not_used):
            label = f"$A_{sens[1]}$" + r"$_,$" + f"$_{sens[i]}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        elif (i!=0) & (i!=len(x)-1):
            ax.scatter(x[i], y[i], marker="+", s=150, color="grey", alpha=0.4)
    ax.set_xlabel("Unfairness")
    ax.set_ylabel("Performance")
    ax.set_xlim((np.min(x)-np.min(x)/10-np.max(x)/10,
                np.max(x)+np.min(x)/10+np.max(x)/10))
    ax.set_ylim((np.min(y)-np.min(y)/100-np.max(y)/100,
                np.max(y)+np.min(y)/100+np.max(y)/100))
    ax.set_title("Exact fairness")
    ax.legend(loc="lower left")
    return ax


def _fair_customized_arrow_plot(unfs_list: list[dict[str, np.ndarray]],
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
            fair_arrow_plot(unfs_list[i], performance_list[i],
                            permutations=True, final_model=False)
        elif i == len(unfs_list)-1:
            fair_arrow_plot(unfs_list[i], performance_list[i],
                            permutations=True, base_model=False)
        else:
            fair_arrow_plot(unfs_list[i], performance_list[i], permutations=True,
                            base_model=False, final_model=False)
    return ax


def fair_multiple_arrow_plot(sensitive_features_calib: np.ndarray,
                             sensitive_features_test: np.ndarray,
                             y_calib: np.ndarray,
                             y_test: np.ndarray,
                             y_true_test: np.ndarray,
                             epsilon: Optional[float] = None,
                             metric: Callable = mean_squared_error) -> plt.Axes:
    """
    Plot arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness for different permutations.

    Parameters
    ----------
    sensitive_features_calib : numpy.ndarray
        Sensitive features for calibration.
    sensitive_features_test : numpy.ndarray
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

    Returns
    -------
    matplotlib.axes.Axes
        Arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness for different permutations.

    Note
    ----
    This function uses a global variable `ax` for plotting, ensuring compatibility with external code.
    """
    permut_y_fair_dict = calculate_perm_wasserstein(
        y_calib, sensitive_features_calib, y_test, sensitive_features_test, epsilon=epsilon)
    all_combs_sensitive_features_test = permutations_columns(
        sensitive_features_test)
    unfs_list = unfairness_permutations(
        permut_y_fair_dict, all_combs_sensitive_features_test)
    performance_list = performance_permutations(
        y_true_test, permut_y_fair_dict, metric=metric)
    return _fair_customized_arrow_plot(unfs_list, performance_list)
