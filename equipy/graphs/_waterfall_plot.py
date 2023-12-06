import matplotlib.pyplot as plt
import numpy as np

def _set_colors(substraction_list):
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


def _add_bar_labels(values, pps, ax):
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


def _add_doted_points(ax, values):
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


def _add_legend(pps, distance, hatch=False):
    """
    Add dotted lines at the top of each bar in a bar plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes on which the bars are plotted.
    values : list
        A list of numerical values representing the heights of the bars.

    Returns
    -------
    matplotlib.axes.Axes
        LineCollection object representing the dotted lines added at the top of each bar in the plot.
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


def _values_to_distance(values):
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


def fair_waterfall_plot(unfs_exact, unfs_approx=None):
    """
    Generate a waterfall plot illustrating the sequential fairness in a model.

    Parameters
    ----------
    unfs_exact : dict
        Dictionary containing fairness values for each step in the exact fairness scenario.
    unfs_approx : dict, optional
        Dictionary containing fairness values for each step in the approximate fairness scenario. Default is None.

    Returns
    -------
    matplotlib.axes.Axes
        The Figure object representing the waterfall plot.

    Notes
    -----
    The function creates a waterfall plot with bars representing the fairness values at each step.
    If both exact and approximate fairness values are provided, bars are color-coded and labeled accordingly.
    The legend is added to distinguish between different bars in the plot.
    """

    fig, ax = plt.subplots()

    handles = []

    leg = tuple(unfs_exact.keys()) + ('Final Model',)
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
        0.5, 0), ncol=3, fancybox=True)

    _add_bar_labels(tuple(base_exact)
                    if unfs_approx is None else tuple(base_approx), pps, ax)
    _add_doted_points(ax, tuple(base_exact)
                      if unfs_approx is None else tuple(base_approx))
    ax.set_ylabel(f'Unfairness in $A_{tuple(unfs_exact.keys())[-1]}$')
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f'Sequential ({"approximate" if unfs_approx is None else "exact"}) fairness: $A_{tuple(unfs_exact.keys())[-1]}$ result')
    plt.show()
