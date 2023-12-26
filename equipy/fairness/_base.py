"""Base class containing all necessary calculations to make predictions fair."""

from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
from ..metrics._fairness_metrics import EQF


class BaseHelper():
    """
    Base class providing helper methods for Wasserstein distance-based fairness adjustment.

    Attributes
    ----------
    ecdf : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality.
    eqf : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality.

    Notes
    -----
    This base class provides essential methods for Wasserstein distance-based fairness adjustment. It includes
    methods for modality extraction, localization of modalities in the input data, weight calculation, and ECDF/EQF 
    estimation with random noise.
    """

    def __init__(self):
        self.ecdf = {}
        self.eqf = {}

        self.weights = {}

    def _get_modalities(self, sensitive_feature: np.ndarray) -> set:
        """
        Get unique modalities from the input sensitive attribute array.

        Parameters
        ----------
        sensitive_feature : array-like, shape (n_samples,)
            Input samples representing the sensitive attributes.

        Returns
        -------
        set
            Set of modalities present in the input sensitive attribute array.
        """
        return set(sensitive_feature)

    def _get_location_modalities(self, sensitive_feature: np.ndarray) -> dict[str, np.ndarray]:
        """
        Get the indices of occurrences for each modality in the input sensitive attribute array.

        Parameters
        ----------
        sensitive_feature : np.ndarray, shape (n_samples,)
            Input sample representing the sensitive attribute.

        Returns
        -------
        dict
            Dictionary where keys are modalities and values are arrays containing their indices.
        """
        location_modalities = {}
        for modality in self._get_modalities(sensitive_feature):
            location_modalities[modality] = np.where(
                sensitive_feature == modality)[0]
        return location_modalities

    def _compute_weights(self, sensitive_feature: np.ndarray) -> dict[str, float]:
        """
        Calculate weights (probabilities) for each modality based on their occurrences.

        Parameters
        ----------
        sensitive_feature : np.ndarray, shape (n_samples,)
            Input samples representing the sensitive attribute.

        Returns
        -------
        dict
            Dictionary where keys are modalities and values are their corresponding weights.
        """
        location_modalities = self._get_location_modalities(sensitive_feature)
        for modality in self._get_modalities(sensitive_feature):
            self.weights[modality] = len(
                location_modalities[modality])/len(sensitive_feature)
        return self.weights

    def _estimate_ecdf_eqf(self, y: np.ndarray, sensitive_feature: np.ndarray, sigma: float) -> tuple(dict[str, float]):
        """
        Estimate ECDF and EQF for each modality, incorporating random noise within [-sigma, sigma].

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values corresponding to the sensitive attribute array.
        sensitive_feature : np.ndarray, shape (n_samples,)
            Input samples representing the sensitive attribute.
        sigma : float
            Standard deviation of the random noise added to the data.

        Returns
        -------
        dict, dict
            Dictionaries where keys are sensitive features and values are their ecdf and eqf.
        """
        location_modalities = self._get_location_modalities(sensitive_feature)
        eps = np.random.uniform(-sigma, sigma, len(y))
        for modality in self._get_modalities(sensitive_feature):
            self.ecdf[modality] = ECDF(y[location_modalities[modality]] +
                                       eps[location_modalities[modality]])
            self.eqf[modality] = EQF(
                y[location_modalities[modality]]+eps[location_modalities[modality]])
        return self.ecdf, self.eqf

    def _get_correction(self, mod: str, y_with_noise: np.ndarray, location_modalities: dict[str, np.ndarray], modalities_test: set) -> float:
        """
        Calculate correction of y.

        Parameters
        ----------
        mod : str
            The current modality for which the correction is calculated.
        y_with_noise : np.ndarray
            y plus a random noise.
        location_modalities : dict
            A dictionary mapping modalities to their locations.
        modalities_test : set
            Set of modalities for which correction is calculated.

        Returns
        -------
        float
            The correction value.
        """
        correction = 0
        for _mod in modalities_test:
            correction += self.weights[_mod] * self.eqf[_mod](
                self.ecdf[mod](y_with_noise[location_modalities[mod]]))
        return correction

    def _fair_y_values(self, y: np.ndarray, sensitive_feature: np.ndarray, modalities_test: list) -> np.ndarray:
        """
        Apply fairness correction to input values.

        Parameters
        ----------
        y : np.ndarray
            Input values.
        sensitive_features : np.ndarray, shape (n_samples, n_sensitive_features)
            The test samples representing multiple sensitive attributes.
        modalities_test : list
            List of modalities for correction.

        Returns
        -------
        np.ndarray
            Fair values after applying correction.
        """
        location_modalities = self._get_location_modalities(sensitive_feature)
        y_fair = np.zeros_like(y)
        eps = np.random.uniform(-self.sigma, self.sigma, len(y))
        y_with_noise = y + eps
        for mod in modalities_test:
            y_fair[location_modalities[mod]] += self._get_correction(
                mod, y_with_noise, location_modalities, modalities_test)
        return y_fair
