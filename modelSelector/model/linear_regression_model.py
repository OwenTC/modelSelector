"""Functionality for model selection for linear regression models."""
import numpy as np
from sklearn.linear_model import LinearRegression

from .base import SalientModelBase


class LinearRegressionModel(SalientModelBase):
    """LinearRegressionModel, for a given id."""

    model: LinearRegression = None
    sigma: float = 0  # standard deviation of the model fit
    rss: float = float("-inf")  # residual sum of squares

    def fit(self, data_x: np.array, data_y: np.array, **kwargs):
        """
        Fits the model for given data.

        Parameters:
        data_x: np.array
            The data for the explanatory variables.
            Here all explanatory variables, even the ones not in the model, are expected!!!!
            Dimensionality: (n_data, n_params_full)
        data_y: np.array
            The data for the predictive variables. dimensionality: (n_data, ) or (n_data, 1)
        """
        data_x = data_x[:, self.parameter_indices_free]

        if self.n_params != data_x.shape[1]:
            raise ValueError("Inconsistent dimensions regarding number of parameters.")

        self.n_data = data_x.shape[0]
        if len(data_y) != self.n_data:
            raise ValueError("Inconsistent dimensions regarding number of data points.")

        if self.n_params == 0:
            self.model = None
            self.rss = np.linalg.norm(data_y) ** 2
            self.sigma = np.sqrt(self.rss/self.n_data)
            self.nll = (
                self.n_data * np.log(self.sigma)
                + 1 / 2 * (self.rss / self.sigma) ** 2
            )
            return

        self.model = LinearRegression()
        self.model.fit(data_x, data_y)

        # Slope Hessian of half the RSS, with the intercept optimized out.
        self._hessian_diagonal = self.n_data * np.var(data_x, axis=0, dtype=np.float64)

        # compute sigma and nll
        self.rss = np.linalg.norm(self.model.predict(data_x) - data_y) ** 2

        self.sigma = np.sqrt(self.rss/self.n_data)

        self.nll = (
            self.n_data * np.log(self.sigma)
            + 1 / 2 * (self.rss / self.sigma) ** 2
        )

    def get_submodel_saliencies(self, saliency: str = "coefficient") -> list[tuple[int, float]]:
        """
        Return normalized saliencies paired with the original feature indices.

        Use ``coefficient`` for |delta_i| or ``hessian`` for H_ii * |delta_i|.
        H_ii is the sum of squared centered feature values, the Hessian diagonal
        of half the residual sum of squares after optimizing the intercept.
        If all scores are zero, assign equal saliency to each active feature.
        """
        if saliency not in ("coefficient", "hessian"):
            raise ValueError("saliency must be 'coefficient' or 'hessian'.")
        if self.n_params == 0:
            return []

        scores = np.abs(np.atleast_2d(self.model.coef_)).sum(axis=0)
        if saliency == "hessian":
            scores = scores * self._hessian_diagonal

        total = scores.sum()
        if total == 0:
            return [(index, 1.0 / self.n_params) for index in self.parameter_indices_free]
        return list(zip(self.parameter_indices_free, (scores / total).tolist()))
