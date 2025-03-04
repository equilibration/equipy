"""
Representation of sequential gain in fairness.
"""

# Authors: Agathe F, Suzie G, Francois H, Philipp R, Arthur C
# License: BSD 3 clause
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import Union, Optional
from ..metrics._fairness_metrics import unfairness_dict
from ..fairness._wasserstein import MultiWasserstein


def _set_colors(substraction_list: list[float]) -> list[str]:
    """
    Assign colors to bars based on the values in the subtraction_list.

    Parameters
    ----------
    subtraction_list : list
        A list of numerical values representing the differences between two sets.

    Returns
    -------
    list
        A list of color codes corresponding to each value in subtraction_list.

    Notes
    -----
    - The color 'tab:orange' is assigned to positive values,
      'tab:green' to non-positive values, and 'tab:grey' to the first and last positions.
    """

    bar_colors = ['tab:grey']
    for i in range(1, len(substraction_list)-1):
        if substraction_list[i] > 0:
            bar_colors.append('tab:orange')
        else:
            bar_colors.append('tab:green')
    bar_colors.append('tab:grey')

    return bar_colors


def _add_bar_labels(values: list[float], pps: list[plt.bar], ax: plt.Axes) -> plt.Axes:
    """
    Add labels to the top of each bar in a bar plot.

    Parameters
    ----------
    values : list
        A list of numerical values representing the heights of the bars.
    pps : list
        A list of bar objects returned by the bar plot.
    ax : matplotlib.axes.Axes
        The Axes on which the bars are plotted.

    Returns
    -------
    matplotlib.axes.Axes
        Text object representing the labels added to the top of each bar in the plot.
    """

    true_values = values + (values[-1],)

    for i, p in enumerate(pps):
        height = true_values[i]
        ax.annotate('{}'.format(height),
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    return ax


def _add_doted_points(ax: plt.Axes, values: np.ndarray) -> plt.Axes:
    """
    Add dotted lines at the top of each bar in a bar plot.

    Parameters
    ----------
    ax : numpy.ndarray
        The Axes on which the bars are plotted.

    values : numpy.ndarray
        An array of numerical values representing the heights of the bars.

    Returns
    -------
    matplotlib.axes.Axes
        The dotted lines at the top of each bar in a bar plot

    This function adds dotted lines at the top of each bar in a bar plot, corresponding to the height values.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> values = np.array([10, 15, 7, 12, 8])
    >>> add_dotted_lines(ax, values)
    >>> plt.show()
    """
    for i, v in enumerate(values):
        ax.plot([i+0.25, i+1.25], [v, v],
                linestyle='--', linewidth=1.5, c='grey')
    return ax


def _add_legend(pps: list[plt.bar], distance: Union[np.ndarray, list], hatch: bool = False) -> list[plt.bar]:
    """
    Add legend labels to the bar plot based on the distances.

    Parameters
    ----------
    pps : List[plt.bar]
        List of bar objects.
    distance : np.ndarray or list
        Array or list of numerical values representing distances.
    hatch : bool, optional
        If True, uses hatching for the legend labels. Defaults to False.

    Returns
    -------
    List[plt.bar]
        List of bar objects with legend labels added.
    """
    used_labels = set()
    for i, bar in enumerate(pps):
        if i == 0 or i == len(pps)-1:
            continue

        if hatch:
            label = 'Net Loss (if exact)' if distance[i] < 0 else 'Net Gain (if exact)'
        else:
            label = 'Net Loss' if distance[i] < 0 else 'Net Gain'

        if label not in used_labels:
            bar.set_label(label)
            used_labels.add(label)
    return pps


def _values_to_distance(values: list[float]) -> list[float]:
    """
    Convert a list of values to a list of distances between consecutive values.

    Parameters
    ----------
    values : list
        A list of numerical values.

    Returns
    -------
    list
        A list of distances between consecutive values.

    Notes
    -----
    This function calculates the differences between consecutive values in the input list, returning a list
    of distances. The last element in the list is the negation of the last value in the input list.
    """
    arr = np.array(values)
    arr = arr[1:] - arr[:-1]
    distance = list(arr) + [-values[-1]]
    return distance

def fair_waterfall_plot(sensitive_features_calib: np.ndarray,
                        sensitive_features_test: np.ndarray,
                        y_calib: np.ndarray,
                        y_test: np.ndarray,
                        epsilon: Optional[float] = None
                        ) -> plt.Axes:
    """
    Generate a waterfall plot illustrating the sequential fairness in a model.

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
    epsilon : float, optional, default = None
        Epsilon value for calculating Wasserstein distance

    Returns
    -------
    matplotlib.axes.Axes
        The Figure object representing the waterfall plot.

    Notes
    -----
    The function creates a waterfall plot with bars representing the fairness values at each step.
    If both exact and approximate fairness values are provided, bars are color-coded and labeled accordingly.
    The legend is added to distinguish between different bars in the plot.

    Examples
    --------
    >>> from sklearn.metrics import f1_score
    >>> sensitive_features_calib = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue'], 'nb_child': [1, 2, 0, 2]})
    >>> sensitive_features_test = pd.DataFrame({'color': ['blue', 'blue', 'blue', 'green'], 'nb_child': [3, 2, 1, 2]})
    >>> y_calib = np.array([0.6, 0.43, 0.32, 0.8])
    >>> y_test = np.array([0.8, 0.35, 0.23, 0.2])
    >>> y_true_test = np.array(['no', 'no', 'yes', 'no'])
    >>> fair_waterfall_plot(sensitive_features_calib, sensitive_features_test, y_calib, y_test, epsilon=[0, 0.5])
    
    """
    
    exact_wst = MultiWasserstein()
    exact_wst.fit(y_calib, sensitive_features_calib)
    y_final_fair = exact_wst.transform(y_test, sensitive_features_test)
    y_sequential_fair = exact_wst.get_sequential_fairness()
    init_unfs_exact = unfairness_dict(y_sequential_fair, sensitive_features_test)
    unfs_exact = {key: round(value, 4) for key, value in init_unfs_exact.items()}
    unfs_approx = None

    if epsilon is not None:
        approx_wst = MultiWasserstein()
        approx_wst.fit(y_calib, sensitive_features_calib)
        approx_y_final_fair = approx_wst.transform(y_test, sensitive_features_test, epsilon=epsilon)
        approx_y_sequential_fair = approx_wst.get_sequential_fairness()
        init_unfs_approx = unfairness_dict(approx_y_sequential_fair, sensitive_features_test)
        unfs_approx = {key: round(value, 4) for key, value in init_unfs_approx.items()}
    
    fig, ax = plt.subplots()

    sens = list(unfs_exact.keys())

    labels = []

    labels = [s + '-fair' for s in sens[1:]]
    leg = ('Base model',) + tuple(labels) + ('Final model',)
    base_exact = list(unfs_exact.values())
    values_exact = [0] + base_exact
    distance_exact = _values_to_distance(values_exact)

    if unfs_approx is not None:

        base_approx = list(unfs_approx.values())
        values_approx = [0] + base_approx
        distance_approx = _values_to_distance(values_approx)

        # waterfall for gray hashed color
        direction = np.array(distance_exact) > 0

        values_grey = np.zeros(len(values_exact))
        values_grey[direction] = np.array(values_approx)[direction]
        values_grey[~direction] = np.array(values_exact)[~direction]

        distance_grey = np.zeros(len(values_exact))
        distance_grey[direction] = np.array(
            values_exact)[direction] - np.array(values_approx)[direction]
        distance_grey[~direction] = np.array(
            values_approx)[~direction] - np.array(values_exact)[~direction]

        # waterfall for exact fairness
        pps0 = ax.bar(leg, distance_exact, color='w', edgecolor=_set_colors(
            distance_exact), bottom=values_exact, hatch='//')

        _add_legend(pps0, distance_exact, hatch=True)

        ax.bar(leg, distance_grey, color='w', edgecolor="grey",
               bottom=values_grey, hatch='//', label='Remains')

        # waterfall for approx. fairness
        pps = ax.bar(leg, distance_approx, color=_set_colors(
            distance_approx), edgecolor='k', bottom=values_approx, label='Baseline')
        _add_legend(pps, distance_approx)

    else:
        # waterfall for exact fairness
        pps = ax.bar(leg, distance_exact, color=_set_colors(
            distance_exact), edgecolor='k', bottom=values_exact, label='Baseline')
        _add_legend(pps, distance_exact)

    fig.legend(loc='upper center', bbox_to_anchor=(
        0.5, 0), ncol=4, fancybox=True)

    _add_bar_labels(tuple(base_exact)
                    if unfs_approx is None else tuple(base_approx), pps, ax)
    _add_doted_points(ax, tuple(base_exact)
                      if unfs_approx is None else tuple(base_approx))
    ax.set_ylabel('Total unfairness')
    ax.set_ylim(0, base_exact[0]+base_exact[0]/10)
    #ax.set_title(
    #    f'Sequential ({"exact" if unfs_approx is None else "approximate"}) fairness: $A_{tuple(unfs_exact.keys())[-1][-1]}$ result')
    ax.set_title(
        f'Sequential ({"exact" if unfs_approx is None else "approximate"}) fairness')
    return fig, ax

