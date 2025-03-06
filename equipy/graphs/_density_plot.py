"""
Representation of the probability distribution of predictions as a function of the value of the sensitive attribute. 
"""

# Authors: Agathe F, Suzie G, Francois H, Philipp R, Arthur C
# License: BSD 3 clause
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Union
from ..fairness._wasserstein import MultiWasserstein
from itertools import product
from scipy.stats import beta

def beta_kernel(data_points: np.ndarray,
                x_grid: np.ndarray,
                bw: Optional[float] = 0.01) -> np.ndarray:
    """
    Computes the density of classification scores using Beta kernel.
    Parameters
    ----------
    data_points : numpy.ndarray
        Classification scores from which KDE is computed. Between 0 and 1.
    x_grid : numpy.ndarray
        Points for which density is evaluated.
    bw : float, optional, default = 0.01
        Bandwidth parameter.
    Returns
    -------
    np.ndarray
        The density function estimated using Beta kernel at points of x_grid.
    """
    kde = []
    for x in x_grid:
        kde.append(beta.pdf(data_points, x / bw + 1, (1 - x) / bw + 1).mean())
    kde = np.asarray(kde)
    return(kde)

def fair_density_plot(sensitive_features_calib: Union[np.ndarray, pd.DataFrame],
                      sensitive_features_test: Union[np.ndarray, pd.DataFrame],
                      y_calib: np.ndarray,
                      y_test: np.ndarray,
                      epsilon: Optional[float] = None,
                      figsize= (26, 18)) -> plt.Axes:
    """
    Visualizes the distribution of predictions based on different sensitive features using kernel density estimates (KDE).

    Parameters
    ----------
    sensitive_features_calib : Union[np.ndarray, pd.DataFrame]
        Sensitive features for calibration.
    sensitive_features_test : Union[np.ndarray, pd.DataFrame]
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
    
    if isinstance(sensitive_features_calib, np.ndarray):
            if len(sensitive_features_calib.shape) == 1:
                sensitive_features_calib = sensitive_features_calib.reshape(-1, 1)
            sensitive_features_calib = pd.DataFrame(
                sensitive_features_calib, columns=[f"sens{i+1}" for i in range(sensitive_features_calib.shape[1])]
                )
    if isinstance(sensitive_features_test, np.ndarray):
            if len(sensitive_features_test.shape) == 1:
                sensitive_features_test = sensitive_features_test.reshape(-1, 1)
            sensitive_features_test = pd.DataFrame(
                sensitive_features_test, columns=[f"sens{i+1}" for i in range(sensitive_features_test.shape[1])]
                )
    
    type = 'regression'
    if np.all((0 <= y_test) & (y_test <= 1)):
        type = 'classification'
        nb_points = np.min([1000, np.max([100, len(y_test)*3])])
        x_grid = np.linspace(0, 1, nb_points)
    
    exact_wst = MultiWasserstein()
    exact_wst.fit(y_calib, sensitive_features_calib)
    y_final_fair = exact_wst.transform(y_test, sensitive_features_test, epsilon=epsilon)
    y_sequential_fair = exact_wst.get_sequential_fairness()
    n = sensitive_features_test.shape[1]+1

    if len(sensitive_features_test.columns) == 1:
        fig, axes = plt.subplots(nrows=n-1, ncols=n, figsize=figsize, constrained_layout=True)
        fig.suptitle('Group-wise model response distribution', fontsize=10)
        modalities = {}
        x_axes = {}
        sensitive_features_test.reset_index(drop=True, inplace=True)
        for i, key in enumerate(y_sequential_fair.keys()):
            if key == 'Base model':
                x_axes[key] = 'Base model predictions'
            else:
                x_axes[key] = f'Fair predictions in {key}'
            df = pd.DataFrame()
            df['Prediction'] = y_sequential_fair[key]
            df = pd.concat([df, sensitive_features_test], axis=1)
            col = sensitive_features_test.columns[0]
            modalities[col] = df[col].unique()
            for mod in modalities[col]:
                subset_data = df[df[col] == mod]
                #sns.kdeplot(
                #    subset_data['Prediction'], label=f'{mod}', fill=True, alpha=0.2, ax=axes[i])
                if type=='regression':
                    sns.kdeplot(
                        subset_data['Prediction'], label=f'{mod}', fill=True, alpha=0.2, ax=axes[i])
                else:
                    kde = beta_kernel(np.array(subset_data['Prediction']), x_grid)
                    sns.lineplot(x = x_grid, y = kde, label=f'{mod}', ax=axes[i])
                    axes[i].fill_between(x_grid, kde, alpha=0.2)
            axes[i].legend(title=col, fontsize=8, title_fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
            axes[i].set_xlabel(x_axes[key], fontsize=8)
            axes[i].set_ylabel('Density', fontsize=8)
            axes[i].xaxis.set_tick_params(labelsize=8)
            axes[i].yaxis.set_tick_params(labelsize=8)
    else:
        fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize, constrained_layout=True)
        fig.suptitle('Group-wise model response distribution', fontsize=30)
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
                    #sns.kdeplot(
                    #    subset_data['Prediction'], label=f'{mod}', fill=True, alpha=0.2, ax=axes[j, i])
                    if type == 'regression':
                        sns.kdeplot(
                            subset_data['Prediction'], label=f'{mod}', fill=True, alpha=0.2,
                            ax=axes[j, i]
                            )
                    else:
                        kde = beta_kernel(np.array(subset_data['Prediction']), x_grid)
                        sns.lineplot(x = x_grid, y = kde, label=f'{mod}', ax=axes[j, i])
                        axes[j, i].fill_between(x_grid, kde, alpha=0.2)
                axes[j, i].legend(title=col, fontsize=14, title_fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
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
                    #sns.kdeplot(
                    #    subset_data['Prediction'], label=perm_str, fill=True, alpha=0.2,
                    #    ax=axes[sensitive_features_test.shape[1], i])
                    if type == 'regression':
                        sns.kdeplot(
                            subset_data['Prediction'], label=perm_str, fill=True, alpha=0.2,
                            ax=axes[sensitive_features_test.shape[1], i]
                            )
                    else:
                        kde = beta_kernel(np.array(subset_data['Prediction']), x_grid)
                        sns.lineplot(x = x_grid, y = kde, label=perm_str, ax=axes[sensitive_features_test.shape[1], i])
                        axes[sensitive_features_test.shape[1], i].fill_between(x_grid, kde, alpha=0.2)
            axes[sensitive_features_test.shape[1], i].legend(title='Intersection', fontsize=20, title_fontsize=20,
                                                             loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
            axes[sensitive_features_test.shape[1], i].set_xlabel(x_axes[key], fontsize=20)
            axes[sensitive_features_test.shape[1], i].set_ylabel('Density', fontsize=20)
            axes[sensitive_features_test.shape[1], i].xaxis.set_tick_params(labelsize=20)
            axes[sensitive_features_test.shape[1], i].yaxis.set_tick_params(labelsize=20)
    return fig, axes
