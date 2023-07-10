from typing import Type, Union, List, Optional

import numpy as np
from scipy.optimize import minimize


def inherit_docstrings(cls: Type) -> Type:
    """
    Decorator to copy docstrings from base class to derived class methods.

    :param cls: The class to decorate.
    :return: The decorated class.
    """
    for name, method in vars(cls).items():
        if method.__doc__ is None:
            for parent in cls.__bases__:
                parent_method = getattr(parent, name)
                if parent_method.__doc__ is not None:
                    method.__doc__ = parent_method.__doc__
                break
    return cls


class Distribution:
    def __init__(
        self,
    ):
        """Initialize a distribution object."""
        pass

    def loss(
        self,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        pass

    def grad(
        self,
        y: np.ndarray,
        z: np.ndarray,
        X: np.ndarray,
        j: int,
    ) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param X: The input training data for the model as a numpy array.
        :param j: The parameter dimension to compute the gradient for (default=0).
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        pass

    def mle(
        self,
        y: np.ndarray,
        z: Optional[np.ndarray] = None,
    ):
        """
        Calculates the maximum likelihood estimate for the parameter.

        :param y: The target values.
        :param z: The current parameter estimates.
        :return: The maximum likelihood estimate for the parameter.
        """
        if z is None:
            z = np.zeros(y.shape[0])
        z_opt = minimize(
            fun=lambda z0: self.loss(y=y, z=z + z0).sum(),
            x0=0,
        )[
            "x"
        ][0]
        return z_opt

    def glm(
        self, y: np.ndarray, X: np.ndarray, z: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculates the initial parameter estimates as a GLM.

        :param y: The target values.
        :param X: The input training data for the model as a numpy array.
        :param z: The current parameter estimates.
        :return: The parameter estimates for a GLM.
        """
        if z is None:
            z = np.zeros(X.shape[0])
        intercept_and_beta_opt = minimize(
            fun=lambda intercept_and_beta: self.loss(
                y=y, z=z + intercept_and_beta[0] + X @ intercept_and_beta[1:, None]
            ).sum(),
            x0=np.zeros(X.shape[1] + 1),
        )["x"]
        intercept = intercept_and_beta_opt[0]
        beta = intercept_and_beta_opt[1:]
        return intercept, beta

    def opt_step(
        self,
        y: np.ndarray,
        z: np.ndarray,
        X: np.ndarray,
        g_0: float,
        j: int,
    ):
        """
        Numerically optimize the step size for the data in specified dimension

        :param y: Target values.
        :param z: Current parameter estimates.
        :param X: Input training data.
        :param j: Index of the dimension to optimize.
        :param g_0: Initial guess for the optimal step size. Default is 0.
        :return: The optimal step size.
        """
        step_opt = minimize(
            fun=lambda step: self.loss(y=y, z=z + X[:, j] * step).sum(),
            jac=lambda step: self.grad(y=y, z=z + X[:, j] * step, X=X, j=j).sum(),
            x0=g_0,
        )["x"][0]
        return step_opt


@inherit_docstrings
class NormalDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a normal distribution object. Parameterization: z = mu"""
        super().__init__()

    def loss(
        self,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return (y - z) ** 2

    def grad(
        self,
        y: np.ndarray,
        z: np.ndarray,
        X: np.ndarray,
        j: int,
    ) -> np.ndarray:
        return -2 * X[:, j] * (y - z)


class GammaDistribution:
    def __init__(
        self,
    ):
        """Initialize a gamma distribution object. Parameterization: z = np.log(mu)"""
        super().__init__()

    def loss(
        self,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return y * np.exp(-z) + z

    def grad(
        self,
        y: np.ndarray,
        z: np.ndarray,
        X: np.ndarray,
        j: int,
    ) -> np.ndarray:
        return X[:, j] * (1 - y * np.exp(-z))


def initiate_distribution(
    distribution: str = "custom",
) -> Distribution:
    """
    Returns a probability distribution object based on the distribution name.

    :param distribution: A string representing the name of the distribution to create.
    :raises ValueError: If the distribution name is not recognized.
    """
    if distribution == "normal":
        return NormalDistribution()
    raise ValueError(f"Unknown distribution: {distribution}")
