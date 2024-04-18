"""
Representation of the probability distribution of predictions as a function of the value of the sensitive attribute. 
"""

# Authors: Agathe F, Suzie G, Francois H, Philipp R, Arthur C
# License: BSD 3 clause
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional
from ..fairness._wasserstein import MultiWasserstein
from itertools import product


def fair_density_plot(sensitive_features_calib: np.ndarray,
                      sensitive_features_test: np.ndarray,
                      y_calib: np.ndarray,
                      y_test: np.ndarray,
                      epsilon: Optional[float] = None) -> plt.Axes:
    """
    Visualizes the distribution of predictions based on different sensitive features using kernel density estimates (KDE).

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
        The density function for predictions based on different sensitive features and fairness.

    Raises
    ------
    ValueError
        If the input data is not in the expected format.

    Examples
    --------
    >>> sensitive_features_calib = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue'], 'nb_child': [1, 2, 0, 2]})
    >>> sensitive_features_test = pd.DataFrame({'color': ['blue', 'blue', 'blue', 'green'], 'nb_child': [3, 2, 1, 2]})
    >>> y_calib = np.array([0.6, 0.43, 0.32, 0.8])
    >>> y_test = np.array([0.8, 0.35, 0.23, 0.2])
    >>> epsilon = [0, 0.5]
    >>> fair_density_plot(sensitive_features_calib, sensitive_features_test, scores_calib, scores_test, epsilon)
    
    """
    
    exact_wst = MultiWasserstein()
    exact_wst.fit(y_calib, sensitive_features_calib)
    y_final_fair = exact_wst.transform(y_test, sensitive_features_test, epsilon=epsilon)
    y_sequential_fair = exact_wst.get_sequential_fairness()

    n = sensitive_features_test.shape[1]+1
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(26, 18))
    fig.suptitle('Density function sequentally fair', fontsize=40)
    modalities = {}
    x_axes = {}

    mod_permutations = list(product(*[sensitive_features_test[col].unique() for
                                      col in sensitive_features_test.columns]))
    sensitive_features_test.reset_index(drop=True, inplace=True)

    for i, key in enumerate(y_sequential_fair.keys()):
        if key == 'Base model':
            x_axes[key] = 'Base model predictions'
        else:
            x_axes[key] = f'Fair predictions in {key}'
        df = pd.DataFrame()
        df['Prediction'] = y_sequential_fair[key]
        df = pd.concat([df, sensitive_features_test], axis=1)
        for j, col in enumerate(sensitive_features_test.columns):
            modalities[col] = df[col].unique()
            for mod in modalities[col]:
                subset_data = df[df[col] == mod]
                sns.kdeplot(
                    subset_data['Prediction'], label=f'{mod}', fill=True, alpha=0.2, ax=axes[j, i])
            axes[j, i].legend(title=col, fontsize=14, title_fontsize=18)
            axes[j, i].set_xlabel(x_axes[key], fontsize=20)
            axes[j, i].set_ylabel('Density', fontsize=20)
            axes[j, i].xaxis.set_tick_params(labelsize=20)
            axes[j, i].yaxis.set_tick_params(labelsize=20)
        for perm in mod_permutations:
            perm_str = '-'.join(map(str, perm))
            conditions = []
            for col, value in zip(sensitive_features_test.columns, perm):
                conditions.append(df[col] == value)
            subset_data = df
            for condition in conditions:
                subset_data = subset_data.loc[condition]
            if not subset_data.empty:
                sns.kdeplot(
                    subset_data['Prediction'], label=perm_str, fill=True, alpha=0.2,
                    ax=axes[sensitive_features_test.shape[1], i])
        axes[sensitive_features_test.shape[1], i].legend(title='Intersection', fontsize=14, title_fontsize=18)
        axes[sensitive_features_test.shape[1], i].set_xlabel(x_axes[key], fontsize=20)
        axes[sensitive_features_test.shape[1], i].set_ylabel('Density', fontsize=20)
        axes[sensitive_features_test.shape[1], i].xaxis.set_tick_params(labelsize=20)
        axes[sensitive_features_test.shape[1], i].yaxis.set_tick_params(labelsize=20)
    return axes
