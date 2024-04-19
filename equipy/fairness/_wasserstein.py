"""
Main Classes to make predictions fair.

The module structure is as follows:

- The FairWasserstein base Class implements fairness adjustment related to a single sensitive attribute, using Wasserstein distance for both binary classification and regression tasks. In the case of binary classification, this class supports scores instead of classes. For more details, see E. Chzhen, C. Denis, M. Hebiri, L. Oneto and M. Pontil, "Fair Regression with Wasserstein Barycenters" (NeurIPS20).
- MultiWasserstein Class extends FairWasserstein for multi-sensitive attribute fairness adjustment in a sequential framework. For more details, see F. Hu, P. Ratz, A. Charpentier, "A Sequentially Fair Mechanism for Multiple Sensitive Attributes" (AAAI24).
"""

# Authors: Agathe F, Suzie G, Francois H, Philipp R, Arthur C
# License: BSD 3 clause
import numpy as np
from ..utils.checkers import _check_epsilon, _check_epsilon_size, _check_mod, _check_shape, _check_nb_observations
from ._base import BaseHelper
from typing import Optional


class FairWasserstein(BaseHelper):
    """
    Class implementing Wasserstein distance-based fairness adjustment for binary classification tasks.

    Parameters
    ----------
    sigma : float, optional (default=0.0001)
        Standard deviation of the random noise added during fairness adjustment.

    Attributes
    ----------
    sigma : float
        Standard deviation of the random noise added during fairness adjustment.
    modalities_calib : dict
        Dictionary storing modality values obtained from calibration data.
    weights : dict
        Dictionary storing weights (probabilities) for each modality based on their occurrences in calibration data.
    ecdf : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality.
    eqf : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality.
    """

    def __init__(self, sigma: float = 0.0001):
        super().__init__()
        self.sigma = sigma
        self.modalities_calib = None

    def fit(self, y: np.ndarray, sensitive_feature: np.ndarray) -> None:
        """
        Perform fit on the calibration data and save the ECDF, EQF, and weights of the sensitive variable.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The calibration labels.

        sensitive_feature : np.ndarray, shape (n_samples,)
            The calibration samples representing one single sensitive attribute.

        Returns
        -------
        None

        Notes
        -----
        This method computes the ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights for the sensitive variable
        based on the provided calibration data. These computed values are used
        during the transformation process to ensure fairness in predictions.

        Examples
        --------
        >>> wasserstein = FairWasserstein(sigma=0.001)
        >>> y = np.array([0.0, 1.0, 1.0, 0.0])
        >>> sensitive_feature = np.array([1, 2, 0, 2])
        >>> wasserstein.fit(y, sensitive_feature)
        """
        _check_shape(y, sensitive_feature)

        self.modalities_calib = self._get_modalities(sensitive_feature)
        self._compute_weights(sensitive_feature)
        self._estimate_ecdf_eqf(y, sensitive_feature, self.sigma)

    def transform(self, y: np.ndarray, sensitive_feature: np.ndarray, epsilon: float = 0) -> np.ndarray:
        """
        Transform the test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The target values of the test data.

        sensitive_feature : np.ndarray, shape (n_samples,)
            The test samples representing a single sensitive attribute.

        epsilon : float, optional (default=0)
            The fairness parameter controlling the trade-off between fairness and accuracy.
            It represents the fraction of the original predictions retained after fairness adjustment.
            Epsilon should be a value between 0 and 1, where 0 means full fairness and 1 means no fairness constraint.

        Returns
        -------
        y_fair : np.ndarray, shape (n_samples,)
            Fair predictions for the test data after enforcing fairness constraints.

        Notes
        -----
        This method applies Wasserstein distance-based fairness adjustment to the test data
        using the precomputed ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights obtained from the calibration data.
        Random noise within the range of [-sigma, sigma] is added to the test data to ensure fairness.
        The parameter epsilon controls the trade-off between fairness and accuracy,
        with 0 enforcing full fairness and 1 retaining the original predictions.

        References
        ----------
        Evgenii Chzhen, Christophe Denis, Mohamed Hebiri, Luca Oneto and Massimiliano Pontil, "Fair Regression with Wasserstein Barycenters" (NeurIPS20)

        Examples
        --------
        >>> y = np.array([0.05, 0.08, 0.9, 0.9, 0.01, 0.88])
        >>> sensitive_feature = np.array([1, 3, 2, 3, 1, 2])
        >>> wasserstein = FairWasserstein(sigma=0.001)
        >>> wasserstein.fit(y, sensitive_feature)
        >>> y = np.array([0.01, 0.99, 0.98, 0.04])
        >>> sensitive_feature = np.array([3, 1, 2, 3])
        >>> print(wasserstein.transform(y, sensitive_feature, epsilon=0.2))
        [0.26063673 0.69140959 0.68940959 0.26663673]
        """

        _check_epsilon(epsilon)
        _check_shape(y, sensitive_feature)
        modalities_test = self._get_modalities(sensitive_feature)
        _check_mod(self.modalities_calib, modalities_test)

        y_fair = self._fair_y_values(y, sensitive_feature, modalities_test)
        return (1-epsilon)*y_fair + epsilon*y


class MultiWasserstein():
    """
    Class extending FairWasserstein for multi-sensitive attribute fairness adjustment.

    Parameters
    ----------
    sigma : float, optional (default=0.0001)
        Standard deviation of the random noise added during fairness adjustment.

    Attributes
    ----------
    sigma : float
        Standard deviation of the random noise added during fairness adjustment.
    y_fair : dict
        Dictionary storing fair predictions for each sensitive feature.
    modalities_calib_all : dict
        Dictionary storing modality values obtained from calibration data for all sensitive features.
    weights_all : dict
        Dictionary storing weights (probabilities) for each modality based on their occurrences in calibration data
        for all sensitive features.
    ecdf_all : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality
        for all sensitive features.
    eqf_all : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality
        for all sensitive features.
    """

    def __init__(self, sigma: float = 0.0001):
        """
        Initialize the MultiWasserStein instance.

        Parameters
        ----------
        sigma : float, optional (default=0.0001)
            The standard deviation of the random noise added to the data during transformation.

        Returns
        -------
        None
        """

        self.y_fair = {}

        self.modalities_calib_all = {}
        self.weights_all = {}

        self.eqf_all = {}
        self.ecdf_all = {}

        self.sigma = sigma

    def fit(self, y: np.ndarray, sensitive_features: np.ndarray) -> None:
        """
        Perform fit on the calibration data and save the ECDF, EQF, and weights for each sensitive variable.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The calibration labels.

        sensitive_features : np.ndarray, shape (n_samples, n_sensitive_features)
            The calibration samples representing multiple sensitive attributes.

        Returns
        -------
        None

        Notes
        -----
        This method computes the ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights for each sensitive variable
        based on the provided calibration data. These computed values are used
        during the transformation process to ensure fairness in predictions.
        """
        _check_nb_observations(sensitive_features)

        if sensitive_features.ndim == 1:
            sensitive_features = np.reshape(
                sensitive_features, (len(sensitive_features), 1))

        for i, sens in enumerate(sensitive_features.T):
            wasserstein_instance = FairWasserstein(sigma=self.sigma)
            if i == 0:
                y_inter = y

            wasserstein_instance.fit(y_inter, sens)
            self.modalities_calib_all[f'sensitive_feature_{i+1}'] = wasserstein_instance.modalities_calib
            self.weights_all[f'sensitive_feature_{i+1}'] = wasserstein_instance.weights
            self.eqf_all[f'sensitive_feature_{i+1}'] = wasserstein_instance.eqf
            self.ecdf_all[f'sensitive_feature_{i+1}'] = wasserstein_instance.ecdf
            y_inter = wasserstein_instance.transform(y_inter, sens)

    def transform(self, y: np.ndarray, sensitive_features: np.ndarray, epsilon: Optional[list[float]] = None) -> np.ndarray:
        """
        Transform the test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The target values of the test data.

        sensitive_features : np.ndarray shape (n_samples, n_sensitive_features)
            The test samples representing multiple sensitive attributes.

        epsilon : list, shape (n_sensitive_features,), optional (default=None)
            The fairness parameters controlling the trade-off between fairness and accuracy
            for each sensitive feature. If None, no fairness constraints are applied.

        Returns
        -------
        y_fair : np.ndarray, shape (n_samples,)
            Fair predictions for the test data after enforcing fairness constraints.

        Notes
        -----
        This method applies Wasserstein distance-based fairness adjustment to the test data
        using the precomputed ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights obtained from the calibration data.
        Random noise within the range of [-sigma, sigma] is added to the test data to ensure fairness.
        The parameter epsilon is a list, where each element controls the trade-off between fairness and accuracy
        for the corresponding sensitive feature.

        References
        ----------
        François Hu, Philipp Ratz, Arthur Charpentier, "A Sequentially Fair Mechanism for Multiple Sensitive Attributes" (AAAI24)

        Examples
        --------
        >>> wasserstein = MultiWasserStein(sigma=0.001)
        >>> y = np.array([0.6, 0.43, 0.32, 0.8])
        >>> sensitive_features = np.array([['blue', 5], ['blue', 9], ['green', 5], ['green', 9]])
        >>> wasserstein.fit(y, sensitive_features)
        >>> y = [0.8, 0.35, 0.23, 0.2]
        >>> sensitive_features = np.array([['blue', 9], ['blue', 5], ['blue', 5], ['green', 9]])
        >>> epsilon = [0.1, 0.2] 
        >>> fair_predictions = wasserstein.transform(y, sensitive_features, epsilon=epsilon)
        >>> print(fair_predictions)
        [0.7015008  0.37444565 0.37204565 0.37144565]
        """
        if epsilon is None:
            if sensitive_features.ndim == 1:
                epsilon = [0]
            else:
                epsilon = [0]*np.shape(sensitive_features)[1]
        _check_epsilon_size(epsilon, sensitive_features)

        self.y_fair['Base model'] = y

        for i, sens in enumerate(sensitive_features.T):
            wasserstein_instance = FairWasserstein(sigma=self.sigma)
            if i == 0:
                y_inter = y
            wasserstein_instance.modalities_calib = self.modalities_calib_all[
                f'sensitive_feature_{i+1}']
            wasserstein_instance.weights = self.weights_all[f'sensitive_feature_{i+1}']
            wasserstein_instance.eqf = self.eqf_all[f'sensitive_feature_{i+1}']
            wasserstein_instance.ecdf = self.ecdf_all[f'sensitive_feature_{i+1}']
            y_inter = wasserstein_instance.transform(
                y_inter, sens, epsilon[i])
            self.y_fair[f'sensitive_feature_{i+1}'] = y_inter
        return self.y_fair[f'sensitive_feature_{i+1}']
