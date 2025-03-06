"""
Computation of the fairness (i.e. measurement of the similarity in prediction distribution between
different population groups according to their sensitive attributes).
"""

# Authors: Agathe F, Suzie G, Francois H, Philipp R, Arthur C
# License: BSD 3 clause
import numpy as np
import warnings
from scipy.interpolate import interp1d
import numpy as np
from typing import Union
import pandas as pd

# WARNING:You cannot calculate the EQF function of a single value : this means that if
# only one individual has a specific sensitive value, you cannot use the transform function.

def identity(x):
    return x

class EQF:
    """
    Empirical Quantile Function (EQF) Class.

    This class computes the linear interpolation of the empirical quantile function for a given set
    of sample data.

    Parameters
    ----------
    sample_data : array-like
        A 1-D array or list-like object containing the sample data.

    Attributes
    ----------
    interpolater : scipy.interpolate.interp1d
        An interpolation function that maps quantiles to values.
    min_val : float
        The minimum value in the sample data.
    max_val : float
        The maximum value in the sample data.

    Methods
    -------
    __init__(sample_data)
        Initializes the EQF object by calculating the interpolater, min_val, and max_val.
    _calculate_eqf(sample_data)
        Private method to calculate interpolater, min_val, and max_val.
    __call__(value_)
        Callable method to compute the interpolated value for a given quantile.

    Raises
    ------
    ValueError
        If the input value_ is outside the range [0, 1].

    Example 
    -------
    >>> sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> eqf = EQF(sample_data)
    >>> print(eqf([0.2, 0.5, 0.8]))  # Interpolated value at quantiles 0.2, 0.5, and 0.8
    [2.8 5.5 8.2]

    Note
    ----
    - The EQF interpolates values within the range [0, 1] representing quantiles.
    - The input sample_data should be a list or array-like containing numerical values.
    """

    def __init__(self, sample_data: Union[np.ndarray, list[float]]):
        self._calculate_eqf(sample_data)
        if len(sample_data) == 1:
            warnings.warn('One of your sample data contains a single value')

    def _calculate_eqf(self, sample_data: Union[np.ndarray, list[float]]) -> None:
        """
        Calculate the Empirical Quantile Function for the given sample data.

        Parameters
        ----------
        sample_data : array-like
            A 1-D array or list-like object containing the sample data.

        Returns
        -------
        EQF
            An instance of the Empirical Quantile Function (EQF) class.

        Notes
        -----
        The EQF interpolates values within the range [0, 1] representing quantiles.
        The input sample_data should be a list or array-like containing numerical values.
        """
        sorted_data = np.sort(sample_data)
        linspace = np.linspace(0, 1, num=len(sample_data))

        if len(sample_data) == 1:
            linspace = np.linspace(0, 1, num=2)
            self.interpolater = interp1d(
                linspace, [sorted_data[0]]*len(linspace))

        else:
            self.interpolater = interp1d(linspace, sorted_data)

        self.min_val = sorted_data[0]
        self.max_val = sorted_data[-1]

    def __call__(self, value_: float) -> float:
        """
        Compute the interpolated value for a given quantile.

        Parameters
        ----------
        value_ : float
            Array of quantile values between 0 and 1.

        Returns
        -------
        float
            Interpolated value corresponding to the input quantile.

        Raises
        ------
        ValueError
            If the input value_ is outside the range [0, 1].
        """
        try:
            return self.interpolater(value_)
        except ValueError:
            if (not isinstance(value_, np.ndarray)) and (not isinstance(value_, float)) and (not isinstance(value_, int)):
                raise ValueError(
                    'value_ can only be an array, a float or an integer number')
            elif (isinstance(value_, np.ndarray)) and (not (np.issubdtype(value_, np.floating) or np.issubdtype(value_, np.integer))):
                raise ValueError(
                    'value_ should contain only float or integer numbers')
            elif np.any(value_ < 0) or np.any(value_ > 1):
                raise ValueError(
                    'value_ should contain only numbers between 0 and 1')
            else:
                raise ValueError('Error with input value')


def diff_quantile(data1: np.ndarray,
                  data2: np.ndarray,
                  approximate=True) -> float:
    """
    Compute the unfairness between two populations based on their quantile functions. If the option
    approximate is set to False, compute the compute the Wasserstein distance using the POT package 
    which requires an install. Else, determine unfairness as the maximum difference in quantiles
    between the two populations.

    Parameters
    ----------
    data1 : np.ndarray
        The first set of data points.
    data2 : np.ndarray
        The second set of data points.
    approximate : bool
        if False, compute distance using optimal transport map, else use quantile approximation

    Returns
    -------
    float
        The unfairness value between the two populations.

    Example
    -------
    >>> data1 = np.array([5, 2, 4, 6, 1])
    >>> data2 = np.array([9, 6, 4, 7, 6])
    >>> diff = compute_unfairness(data1, data2, n_min=5)
    >>> print(diff)
    3.9797979797979797
    """
    if approximate:
        probs = np.linspace(0.01, 0.99, num=100)
        eqf1 = np.quantile(data1, probs)
        eqf2 = np.quantile(data2, probs)
        unfair_value = np.max(np.abs(eqf1-eqf2))
    else:
        try:
            import ot
        except ModuleNotFoundError:
            print('POT not installed, install before using'\
                  'approximate=False option')
        except Exception as e:
            print(f"Unexpected {e=}, {type(e)=}")
            raise

        n1 = len(data1)  # data1 corresponds to y
        n2 = len(data2)
        # weights of each point of the two distributions
        a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
        M = ot.dist(data1.reshape((n1, 1)), data2.reshape((n2, 1)),
                    metric='euclidean')  # euclidian distance matrix
        M = M/M.max()
        unfair_value = ot.emd2(a, b, M)

    return unfair_value


def unfairness(y: np.ndarray, sensitive_features: Union[np.ndarray, pd.DataFrame], n_min: float = 1000) -> float:
    """
    Compute the unfairness value for a given fair output (y) and multiple sensitive attributes data
    (sensitive_features) containing several modalities. If there is a single sensitive feature,
    it calculates the maximum quantile difference between different modalities of that single
    sensitive feature. If there are multiple sensitive features, it calculates the maximum quantile
    difference for each sensitive feature and then takes the sum of these maximums.

    Parameters
    ----------
    y : np.ndarray
        Predicted (fair or not) outcome variable.
    sensitive_features : Union[np.ndarray, pd.DataFrame]
        Sensitive attribute data.
    n_min : float
        Below this threshold, compute the unfairness based on the Wasserstein distance.

    Returns
    -------
    float
        Unfairness value in the dataset.

    Example
    -------
    >>> y = np.array([5, 0, 6, 7, 9])
    >>> sensitive_features = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue'],
                                           'nb_child': [1, 2, 0, 2]})
    >>> unf = unfairness(y, sensitive_features, n_min=5)
    >>> print(unf)
    6.0
    """
    
    if isinstance(sensitive_features, np.ndarray):
        sensitive_features = pd.DataFrame(
            sensitive_features, columns=[f"sens{i+1}" for i in range(sensitive_features.shape[1])]
            )
    new_list = []
    for col in sensitive_features.columns:
        sensitive_feature = sensitive_features[col]
        modalities = list(sensitive_feature.unique())
        lst_unfairness = []
        for modality in modalities:
            y_modality = y[sensitive_feature == modality]
            lst_unfairness.append(diff_quantile(y, y_modality, n_min))
        new_list.append(max(lst_unfairness))
    unfs = np.sum(new_list)
    return unfs


def unfairness_dict(y_fair_dict: dict[str, np.ndarray],
                    sensitive_features: Union[np.ndarray, pd.DataFrame],
                    n_min: float = 1000) -> dict[str, float]:
    """
    Compute unfairness values for sequentially fair output datasets and multiple sensitive
    attributes datasets.

    Parameters
    ----------
    y_fair_dict : dict
        A dictionary where keys represent sensitive features and values are arrays
        containing the fair predictions corresponding to each sensitive feature.
        Each sensitive feature's fairness adjustment is performed sequentially,
        ensuring that each feature is treated fairly relative to the previous ones.
    sensitive_features : Union[np.ndarray, pd.DataFrame]
        Sensitive attribute data.
    n_min : float
        Below this threshold, compute the unfairness based on the Wasserstein distance.

    Returns
    -------
    dict
        A dictionary containing unfairness values for each level of fairness.
        The level of fairness corresponds to the number of sensitive attributes to which 
        fairness has been applied.

    Example
    -------
    >>> y_fair_dict = {'Base model':np.array([19,39,65]), 'color':np.array([22,40,50]),
                       'nb_child':np.array([28,39,42])}
    >>> sensitive_features = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue'],
                                           'nb_child': [1, 2, 0, 2]})
    >>> unfs_dict = unfairness_dict(y_fair_dict, sensitive_features, n_min=5)
    >>> print(unfs_dict)
    {'Base model': 46.0, 'color': 28.0, 'nb_child': 14.0}
    """
    if isinstance(sensitive_features, np.ndarray):
        sensitive_features = pd.DataFrame(
            sensitive_features, columns=[f"sens{i+1}" for i in range(sensitive_features.shape[1])]
            )

    unfairness_dict = {}
    for key, y_fair in y_fair_dict.items():
        result = unfairness(y_fair, sensitive_features, n_min)
        unfairness_dict[key] = result
    return unfairness_dict
