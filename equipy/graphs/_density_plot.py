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

    Example
    -------
    >>> y = {
            'Base model': [prediction_values],
            'sensitive_feature_1': [prediction_values],
            'sensitive_feature_2': [prediction_values],
            ...
        }
    >>> sensitive_features = [[sensitive_features_of_ind_1_values], [sensitive_feature_of_ind_2_values], ...]
    """

    exact_wst = MultiWasserstein()
    exact_wst.fit(y_calib, sensitive_features_calib)
    y_final_fair = exact_wst.transform(y_test, sensitive_features_test, epsilon=epsilon)
    y_sequential_fair = exact_wst.get_sequential_fairness()

    fig, axes = plt.subplots(nrows=sensitive_features_test.shape[1], ncols=
                         sensitive_features_test.shape[1]+1, figsize=(26, 18))
    fig.suptitle('Density function sequentally fair', fontsize=40)
    axes = axes.flatten(order='F')
    graph = 0
    modalities = {}
    x_axes = {}

    sensitive_features_test.reset_index(drop=True, inplace=True)
    for key in y_sequential_fair.keys():
        if key == 'Base model':
            x_axes[key] = 'Base model predictions'
        else:
            x_axes[key] = f'Fair predictions in {key}'
        df = pd.DataFrame()
        df['Prediction'] = y_sequential_fair[key]
        for col in sensitive_features_test.columns:
            sensitive_feature = sensitive_features_test[col]
            df[col] = sensitive_feature
            modalities[col] = df[col].unique()
            for mod in modalities[col]:
                subset_data = df[df[col] == mod]
                sns.kdeplot(
                    subset_data['Prediction'], label=f'{col}: {mod}', fill=True, alpha=0.2, ax=axes[graph])
            axes[graph].legend(fontsize=15)
            axes[graph].set_xlabel(x_axes[key], fontsize=20)
            axes[graph].set_ylabel('Density', fontsize=20)
            axes[graph].xaxis.set_tick_params(labelsize=20)
            axes[graph].yaxis.set_tick_params(labelsize=20)
            graph += 1
    return axes
