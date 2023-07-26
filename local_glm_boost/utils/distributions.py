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
        w: Union[float, np.ndarray] = 1,
    ) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :param w: The weights of the observations. If `None`, all weights are set to 1.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        pass

    def grad(
        self,
        y: np.ndarray,
        z: np.ndarray,
        w: Union[float, np.ndarray] = 1,
    ) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param w: The weights of the observations. If `None`, all weights are set to 1.
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        pass


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
        w: Union[float, np.ndarray] = 1,
    ) -> np.ndarray:
        return (y - w * z) ** 2

    def grad(
        self,
        y: np.ndarray,
        z: np.ndarray,
        w: Union[float, np.ndarray] = 1,
    ) -> np.ndarray:
        return -2 * (y - w * z)


@inherit_docstrings
class PoissonDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a Poisson distribution object. Parameterization: z = np.log(lambda)"""
        super().__init__()

    def loss(
        self,
        y: np.ndarray,
        z: np.ndarray,
        w: Union[float, np.ndarray] = 1,
    ) -> np.ndarray:
        return w * np.exp(z) - y * z

    def grad(
        self,
        y: np.ndarray,
        z: np.ndarray,
        w: Union[float, np.ndarray] = 1,
    ) -> np.ndarray:
        return w * np.exp(z) - y


@inherit_docstrings
class GammaDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a normal distribution object. Parameterization: z = mu"""
        super().__init__()

    def loss(
        self,
        y: np.ndarray,
        z: np.ndarray,
        w: Union[float, np.ndarray] = 1,
    ) -> np.ndarray:
        return y * np.exp(-z) + w * z

    def grad(
        self,
        y: np.ndarray,
        z: np.ndarray,
        w: Union[float, np.ndarray] = 1,
    ) -> np.ndarray:
        return w - y * np.exp(-z)


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
    if distribution == "gamma":
        return GammaDistribution()
    if distribution == "poisson":
        return PoissonDistribution()
    raise ValueError(f"Unknown distribution: {distribution}")
