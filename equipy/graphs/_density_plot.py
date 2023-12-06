import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def fair_density_plot(y, sensitive_features):
    """
    Visualizes the distribution of predictions based on different sensitive features using kernel density estimates (KDE).

    Parameters
    ----------
    y : dict
        A dictionary containing sequentially fair output datasets.
    sensitive_features : np.ndarray, shape (n_samples, n_sensitive_features)
        The samples representing multiple sensitive attributes.

    Returns
    -------
    matplotlib.figure.Figure
        the density function for predictions based on different sensitive features and fairness.

    Raises
    ------
    ValueError
        If the input data is not in the expected format.

    Plotting Conventions
    --------------------
    - The x-axis represents prediction values, and the y-axis represents density.

    Example
    -------
    >>> y = {
            'Base model': [prediction_values],
            'sensitive_feature_1': [prediction_values],
            'sensitive_feature_2': [prediction_values],
            ...
        }
    >>> sensitive_features = [[sensitive_features_of_ind_1_values], [sensitive_feature_of_ind_2_values], ...]

    Usage
    -----
    fairness_density_plot(y, sensitive_features)
    """

    fig, axes = plt.subplots(nrows=len(sensitive_features.T), ncols=len(
        sensitive_features.T)+1, figsize=(26, 18))
    fig.suptitle('Density function sequentally fair', fontsize=40)

    axes = axes.flatten(order='F')

    graph = 0
    modalities = {}

    for key in y.keys():
        df = pd.DataFrame()
        df['Prediction'] = y[key]
        title = key
        for i, sensitive_feature in enumerate(sensitive_features.T):
            df[f"sensitive_feature_{i+1}"] = sensitive_feature
            modalities[i] = df[f"sensitive_feature_{i+1}"].unique()
            for mod in modalities[i]:
                subset_data = df[df[f'sensitive_feature_{i+1}'] == mod]
                sns.kdeplot(
                    subset_data['Prediction'], label=f'sensitive_feature_{i+1}: {mod}', fill=True, alpha=0.2, ax=axes[graph])
            axes[graph].legend(fontsize=15)
            axes[graph].set_title(title, fontsize=30)
            axes[graph].set_xlabel('Prediction', fontsize=20)
            axes[graph].set_ylabel('Density', fontsize=20)
            axes[graph].xaxis.set_tick_params(labelsize=20)
            axes[graph].yaxis.set_tick_params(labelsize=20)
            graph += 1
