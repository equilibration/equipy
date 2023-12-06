import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from ..utils.permutations._compute_permutations import permutations_columns, calculate_perm_wasserstein
from ..utils.permutations.metrics._fairness_permutations import unfairness_permutations
from ..utils.permutations.metrics._performance_permutations import performance_permutations


def fair_arrow_plot(unfs_dict, performance_dict, permutations=False, base_model=True, final_model=True):
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
    matplotlib.figure.Figure 
        arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness.

    Plotting Conventions
    --------------------
    - Arrows represent different fairness-performance combinations.
    - Axes are labeled for unfairness (x-axis) and performance (y-axis).

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
            sens.append(int(key[9:]))

    for key in performance_dict.keys():
        y.append(performance_dict[key])

    global ax

    if not permutations:
        fig, ax = plt.subplots()

    line = ax.plot(x, y, linestyle="--", alpha=0.25, color="grey")[0]

    for i in range(len(sens)):
        if (i == 0) & (base_model):
            line.axes.annotate(f"Base\nmodel", xytext=(
                x[0]+np.min(x)/20, y[0]), xy=(x[0], y[0]), size=10)
            ax.scatter(x[0], y[0], label="Base model", marker="^", s=100)
        elif i == 1:
            label = f"$A_{sens[i]}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        elif (i == len(x)-1) & (final_model):

            label = f"$A_{1}$" + r"$_:$" + f"$_{i}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="*", s=150)
        elif (i == 2) & (i < len(x)-1):

            label = f"$A_{sens[1]}$" + r"$_,$" + f"$_{sens[i]}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        else:
            ax.scatter(x[i], y[i], marker="+", s=150, color="grey", alpha=0.4)
    ax.set_xlabel("Unfairness")
    ax.set_ylabel("Performance")
    ax.set_xlim((np.min(x)-np.min(x)/10-np.max(x)/10,
                np.max(x)+np.min(x)/10+np.max(x)/10))
    ax.set_ylim((np.min(y)-np.min(y)/10-np.max(y)/10,
                np.max(y)+np.min(y)/10+np.max(y)/10))
    ax.set_title("Exact fairness")
    ax.legend(loc="best")


def _fair_custimized_arrow_plot(unfs_list, performance_list):
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
    matplotlib.figure.Figure
        arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness for each combination.

    Plotting Conventions
    --------------------
    - Arrows represent different fairness-performance combinations for each scenario in the input lists.
    - Axes are labeled for unfairness (x-axis) and performance (y-axis).

    Example Usage
    -------------
    >>> arrow_plot_permutations(unfs_list, performance_list)

    Note
    ----
    This function uses a global variable `ax` for plotting, ensuring compatibility with external code.
    """
    global ax
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


def fair_multiple_arrow_plot(sensitive_features_calib, sensitive_features_test, y_calib, y_test, y_true_test, epsilon=None, test_size=0.3, permutation=True, metric=mean_squared_error):
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
    epsilon : float, optional
        Epsilon value for calculating Wasserstein distance. Defaults to None.
    test_size : float, optional
        Size of the testing set. Defaults to 0.3.
    permutation : bool, optional
        If True, displays permutations of arrows based on input dictionaries. Defaults to True.
    metric : function, optional
        The metric used to evaluate performance. Defaults to mean_squared_error.

    Returns
    -------
    matplotlib.axes.Axes
        Arrows representing the fairness-performance combinations step by step (by sensitive attribute) to reach fairness for different permutations.

    Plotting Conventions
    --------------------
    - Arrows represent different fairness-performance combinations for each permutation.
    - Axes are labeled for unfairness (x-axis) and performance (y-axis).

    Example Usage
    -------------
    >>> custom_fair_arrow_plot(sensitive_features_calib, sensitive_features_test, y_calib, y_test, y_true_test)

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
    _fair_custimized_arrow_plot(unfs_list, performance_list)
