"""
Main Classes to make predictions fair.

The module structure is as follows:

- The FairWasserstein base Class implements fairness adjustment related to a single sensitive attribute, using Wasserstein distance for both binary classification and regression tasks. In the case of binary classification, this class supports scores instead of classes. For more details, see E. Chzhen, C. Denis, M. Hebiri, L. Oneto and M. Pontil, "Fair Regression with Wasserstein Barycenters" (NeurIPS20).
- MultiWasserstein Class extends FairWasserstein for multi-sensitive attribute fairness adjustment in a sequential framework. For more details, see F. Hu, P. Ratz, A. Charpentier, "A Sequentially Fair Mechanism for Multiple Sensitive Attributes" (AAAI24).
"""

# Authors: Agathe F, Suzie G, Francois H, Philipp R, Arthur C
# License: BSD 3 clause
import numpy as np
import pandas as pd
import itertools
from ..utils.checkers import _check_epsilon, _check_epsilon_size, _check_mod, _check_shape, _check_nb_observations, _check_col, _check_unique_mod
from ..metrics._fairness_metrics import identity
from ._base import BaseHelper
from typing import Optional


class FairWasserstein(BaseHelper):
    """
    Class implementing Wasserstein distance-based fairness adjustment for binary classification and regression tasks regarding a single sensitive attribute.

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
        self.columns_calib = None

    def fit(self, y: np.ndarray, sensitive_feature: pd.DataFrame) -> None:
        """
        Perform fit on the calibration data and save the ECDF, EQF, and weights of the sensitive variable.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The calibration labels.

        sensitive_feature : pd.DataFrame, shape (n_samples, 1)
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
        >>> sensitive_feature = pd.DataFrame({'nb_child': [1, 2, 0, 2]})
        >>> wasserstein.fit(y, sensitive_feature)
        """
        _check_shape(y, sensitive_feature)
        _check_unique_mod(sensitive_feature)

        self.modalities_calib = self._get_modalities(sensitive_feature)
        self.columns_calib = sensitive_feature.columns
        self._compute_weights(sensitive_feature)
        self._estimate_ecdf_eqf(y, sensitive_feature, self.sigma)

    def transform(self, y: np.ndarray, sensitive_feature: pd.DataFrame, epsilon: float = 0) -> np.ndarray:
        """
        Transform the test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The target values of the test data.

        sensitive_feature : pd.DataFrame, shape (n_samples, 1)
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
        >>> sensitive_feature = pd.DataFrame({'nb_child': [1, 3, 2, 3, 1, 2]})
        >>> wasserstein = FairWasserstein(sigma=0.001)
        >>> wasserstein.fit(y, sensitive_feature)
        >>> y = np.array([0.01, 0.99, 0.98, 0.04])
        >>> sensitive_feature = pd.DataFrame({'nb_child': [3, 1, 2, 3]})
        >>> print(wasserstein.transform(y, sensitive_feature, epsilon=0.2))
        [0.26063673 0.69140959 0.68940959 0.26663673]
        """

        _check_epsilon(epsilon)
        _check_shape(y, sensitive_feature)
        modalities_test = self._get_modalities(sensitive_feature)
        columns_test = sensitive_feature.columns
        _check_mod(self.modalities_calib, modalities_test)
        _check_col(self.columns_calib, columns_test)

        y_fair = self._fair_y_values(y, sensitive_feature, modalities_test)
        return (1-epsilon)*y_fair + epsilon*y
    
    def fit_transform(self, y_calib: np.ndarray, sensitive_feature_calib:pd.DataFrame, y_test:np.ndarray, sensitive_feature_test:pd.DataFrame, epsilon:float= 0) -> np.ndarray:
        """
        Fit and transform the calibration and test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y_calib : np.ndarray, shape (n_samples,)
            The target values of the calibration data.

        sensitive_feature_calib : pd.DataFrame, shape (n_samples, 1)
            The calibration samples representing a single sensitive attribute.

        y_test : np.ndarray, shape (n_samples,)
            The target values of the test data.

        sensitive_feature_test : pd.DataFrame, shape (n_samples, 1)
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
        >>> y_calib = np.array([0.05, 0.08, 0.9, 0.9, 0.01, 0.88])
        >>> sensitive_feature_calib = pd.DataFrame({'nb_child': [1, 3, 2, 3, 1, 2]})
        >>> y_test = np.array([0.01, 0.99, 0.98, 0.04])
        >>> sensitive_feature_test = pd.DataFrame({'nb_child': [3, 1, 2, 3]})
        >>> wasserstein = FairWasserstein(sigma=0.001)
        >>> print(wasserstein.fit_transform(y_calib, sensitive_feature_calib, y_test, sensitive_feature_test, epsilon=0.2))
        [0.26063673 0.69140959 0.68940959 0.26663673]
        """
        self.fit(y_calib, sensitive_feature_calib)
        return self.transform(y_test, sensitive_feature_test, epsilon)


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
        self.columns_calib_all = None

        self.weights_all = {}

        self.eqf_all = {}
        self.ecdf_all = {}

        self.sigma = sigma

    def fit(self, y: np.ndarray, sensitive_features: pd.DataFrame) -> None:
        """
        Perform fit on the calibration data and save the ECDF, EQF, and weights for each sensitive variable.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The calibration labels.

        sensitive_features : pd.DataFrame, shape (n_samples, n_sensitive_features)
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
        _check_shape(y, sensitive_features)
        self.columns_calib_all = sensitive_features.columns

        y_inter = y.copy()

        for col in sensitive_features.columns:
            if sensitive_features.shape[1] > 1:
                self.modalities_calib_all[col] = {}
                self.weights_all[col] = {}
                self.ecdf_all[col] = {}
                self.eqf_all[col] = {}
                sensitive_filtered = sensitive_features.drop(columns=[col])
                combinations = sensitive_filtered.drop_duplicates().copy()
                combinations['concat'] = combinations.astype(str).agg(''.join, axis=1)
                sensitive_filtered = sensitive_filtered.astype(str).agg(''.join, axis=1)
                for value in combinations['concat']:
                    cond = sensitive_filtered == value
                    intersection = sensitive_features.loc[cond].apply(lambda row: ''.join(row.astype(str)), axis=1)
                    wasserstein_instance = FairWasserstein(sigma = self.sigma)
                    if len(intersection.unique()) == 1:
                        self.modalities_calib_all[col][value] = set(intersection.unique())
                        self.weights_all[col][value] = {intersection.unique()[0]: 1}
                        self.eqf_all[col][value] = {intersection.unique()[0]: identity}
                        self.ecdf_all[col][value] = {intersection.unique()[0]: identity}
                    else:
                        new_sens = pd.DataFrame({'intersection': intersection})
                        wasserstein_instance.fit(y_inter[cond], new_sens)
                        self.modalities_calib_all[col][value] = wasserstein_instance.modalities_calib
                        self.weights_all[col][value] = wasserstein_instance.weights
                        self.eqf_all[col][value] = wasserstein_instance.eqf
                        self.ecdf_all[col][value] = wasserstein_instance.ecdf
                        y_inter[cond] = wasserstein_instance.transform(y_inter[cond], new_sens)
                sensitive_features = sensitive_features.drop(columns=col)
            else:
                wasserstein_instance = FairWasserstein(sigma = self.sigma)
                wasserstein_instance.fit(y_inter, sensitive_features[[col]])
                self.modalities_calib_all[col] = wasserstein_instance.modalities_calib
                self.weights_all[col] = wasserstein_instance.weights
                self.eqf_all[col] = wasserstein_instance.eqf
                self.ecdf_all[col] = wasserstein_instance.ecdf
                y_inter = wasserstein_instance.transform(y_inter, sensitive_features[[col]])
    
    def transform(self, y: np.ndarray, sensitive_features: pd.DataFrame, epsilon: Optional[list[float]] = None) -> np.ndarray:
        """
        Transform the calib and test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            The target values of the test data.

        sensitive_features : pd.DataFrame shape (n_samples, n_sensitive_features)
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
        >>> sensitive_features = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue'], 'nb_child': [1, 2, 0, 2]})
        >>> wasserstein.fit(y, sensitive_features)
        >>> y = np.array([0.8, 0.35, 0.23, 0.2])
        >>> sensitive_features = pd.DataFrame({'color': ['blue', 'blue', 'blue', 'green'], 'nb_child': [2, 2, 1, 2]})
        >>> epsilon = [0.1, 0.2] 
        >>> fair_predictions = wasserstein.transform(y, sensitive_features, epsilon=epsilon)
        >>> print(fair_predictions)
        [0.42483123 0.36412012 0.36172012 0.36112012]
        """
        if epsilon is None:
            if sensitive_features.shape[1] == 1:
                epsilon = [0]
            else:
                epsilon = [0]*sensitive_features.shape[1]
        _check_epsilon_size(epsilon, sensitive_features)
        _check_col(self.columns_calib_all, sensitive_features.columns)

        self.y_fair['Base model'] = y
        y_inter = y.copy()

        for i, col in enumerate(sensitive_features.columns):
            if sensitive_features.shape[1] > 1:
                sens_filtered = sensitive_features.drop(columns=[col])
                combinations = sens_filtered.drop_duplicates().reset_index(drop=True)
                combinations['concat'] = combinations.astype(str).agg(''.join, axis=1)
                sens_filtered = sens_filtered.astype(str).agg(''.join, axis=1)
                for value in combinations['concat']:
                    cond = sens_filtered == value
                    intersection = sensitive_features[cond].apply(lambda row: ''.join(row.astype(str)), axis=1)
                    new_sens = pd.DataFrame({'intersection': intersection})
                    wasserstein_instance = FairWasserstein(sigma = self.sigma)
                    wasserstein_instance.columns_calib = new_sens.columns
                    wasserstein_instance.modalities_calib = self.modalities_calib_all[col][value]
                    wasserstein_instance.weights = self.weights_all[col][value]
                    wasserstein_instance.eqf = self.eqf_all[col][value]
                    wasserstein_instance.ecdf = self.ecdf_all[col][value]
                    y_inter[cond] = wasserstein_instance.transform(y_inter[cond], new_sens, epsilon[i])
                sensitive_features = sensitive_features.drop(columns=col)
            else:
                wasserstein_instance = FairWasserstein(sigma = self.sigma)
                wasserstein_instance.columns_calib = sensitive_features[[col]].columns
                wasserstein_instance.modalities_calib = self.modalities_calib_all[col]
                wasserstein_instance.weights = self.weights_all[col]
                wasserstein_instance.eqf = self.eqf_all[col]
                wasserstein_instance.ecdf = self.ecdf_all[col]
                wasserstein_instance.columns_calib = sensitive_features[[col]].columns
                y_inter = wasserstein_instance.transform(y_inter, sensitive_features[[col]], epsilon[i])                                                                     
            self.y_fair[col] = y_inter.copy()
        return self.y_fair[col]
    
    def fit_transform(self, y_calib:np.ndarray, sensitive_features_calib:pd.DataFrame, y_test:np.ndarray, sensitive_features_test:pd.DataFrame,epsilon: Optional[list[float]] = None)->np.array:
        """
        Fit and transform the calibration and test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y_calib : np.ndarray, shape (n_samples,)
            The calibration labels.

        sensitive_features_calub : pd.DataFrame, shape (n_samples, n_sensitive_features)
            The calibration samples representing multiple sensitive attributes.

        y_test : np.ndarray, shape (n_samples,)
            The target values of the test data.

        sensitive_features_test : pd.DataFrame shape (n_samples, n_sensitive_features)
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
        >>> y_calib = np.array([0.6, 0.43, 0.32, 0.8])
        >>> sensitive_features_calib = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue'], 'nb_child': [1, 2, 0, 2]})
        >>> y_test = [0.8, 0.35, 0.23, 0.2]
        >>> sensitive_features_test = pd.DataFrame({'color': ['blue', 'blue', 'blue', 'green'], 'nb_child': [2, 2, 1, 2]})
        >>> epsilon = [0.1, 0.2] 
        >>> print(wasserstein.fit_transform(y_calib, sensitive_features_calib, y_test, sensitive_features_test, epsilon))
        [0.42483123 0.36412012 0.36172012 0.36112012]
        """
        self.fit(y_calib, sensitive_features_calib)
        return self.transform(y_test, sensitive_features_test, epsilon)

    def get_sequential_fairness(self) -> dict:
        """
        Returns a dictionary containing fair outputs generated at each iteration of the application of the transform method.
        These outputs represent the predictions for the test data after enforcing fairness at each step.

        Returns
        -------
        y_sequential_fair : dict
            A dictionary containing fair predictions for the test data at each iteration of applying the fairness transformations, regarding one sensitive attribute.
        """
        y_sequential_fair = self.y_fair
        return y_sequential_fair
