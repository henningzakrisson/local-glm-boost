from typing import List, Union, Optional

import numpy as np
import pandas as pd

from .utils.distributions import Distribution, initiate_distribution
from .utils.fix_datatype import fix_datatype
from .boosting_tree import LocalBoostingTree


class LocalGLMBooster:
    def __init__(
        self,
        distribution: Union[Distribution, str] = "normal",
        n_estimators: Union[List[int], int] = 100,
        learning_rate: Union[List[float], float] = 0.1,
        min_samples_split: Union[List[int], int] = 2,
        min_samples_leaf: Union[List[int], int] = 1,
        max_depth: Union[List[int], int] = 3,
        glm_init: Union[List[bool], bool] = True,
    ):
        """
        Initialize a LocalGLMBooster model.

        :param distribution: The distribution of the response variable. A Distribution object or a string.
        :param n_estimators: Number of boosting steps. Dimension-wise or global for all coefficients.
        :param learning_rate: Shrinkage factors, which scales the contribution of each tree. Dimension-wise or global for all coefficients
        :param min_samples_split: Minimum number of samples required to split an internal node. Dimension-wise or global for all coefficients.
        :param min_samples_leaf: Minimum number of samples required at a leaf node. Dimension-wise or global for all coefficients.
        :param max_depth: Maximum depths of each decision tree. Dimension-wise or global for all coefficients.
        :param glm_init: Whether to initialize the model with a GLM fit.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.glm_init = glm_init

        if isinstance(distribution, str):
            self.distribution = initiate_distribution(distribution)
        else:
            self.distribution = distribution

        self.p = None
        self.beta0 = None
        self.z0 = None
        self.trees = None
        self.feature_names = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Fit the model to the data.

        :param X: Input data matrix of shape (n, p).
        :param y: True response values for the input data of shape (n,).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        X, y = fix_datatype(X=X, y=y)

        self.p = X.shape[1]
        self._adjust_hyperparameters()
        self.trees = [
            [
                LocalBoostingTree(
                    distribution=self.distribution,
                    max_depth=self.max_depth[j],
                    min_samples_split=self.min_samples_split[j],
                    min_samples_leaf=self.min_samples_leaf[j],
                )
                for _ in range(self.n_estimators[j])
            ]
            for j in range(self.p)
        ]

        self.beta0 = np.zeros(self.p)
        if np.any(self.glm_init):
            self.z0, beta0 = self.distribution.glm(X=X[:, self.glm_init], y=y)
            self.beta0[self.glm_init] = beta0
        else:
            self.z0 = self.distribution.mle(y=y)

        z = self.z0 + (self.beta0.T @ X.T).T.reshape(-1)

        for k in range(max(self.n_estimators)):
            for j in range(self.p):
                if k < self.n_estimators[j]:
                    self.trees[j][k].fit_gradients(X=X, y=y, z=z, j=j)
                    z += self.learning_rate[j] * self.trees[j][k].predict(X) * X[:, j]

        # Re-adjust the initial parameter values
        if np.any(self.glm_init):
            self.z0, beta0 = self.distribution.glm(X=X[:, self.glm_init], y=y)
            self.beta0[self.glm_init] = beta0
        else:
            self.z0 = self.distribution.mle(y=y)

    def _adjust_hyperparameters(self) -> None:
        """Adjust hyperparameters given the new covariate dimensions."""

        def adjust_param(param: str):
            param_value = getattr(self, param)
            if isinstance(param_value, List) or isinstance(param_value, np.ndarray):
                if len(param_value) != self.p:
                    raise ValueError(
                        f"Length of {param} must be equal to the number of covariates."
                    )
            else:
                setattr(self, param, [param_value] * self.p)

        for param in [
            "n_estimators",
            "learning_rate",
            "min_samples_split",
            "min_samples_leaf",
            "max_depth",
            "glm_init",
        ]:
            adjust_param(param)

    def add_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: int,
        z: Optional[np.ndarray] = None,
    ) -> None:
        """
        Updates the current boosting model with one additional tree to the jth coefficient.

        :param X: The training input data, shape (n_samples, n_features).
        :param y: The target values for the training data.
        :param j: Coefficient  to update
        :param z: The current predictions of the model. If None, the predictions are computed from the current model.
        """
        if z is None:
            z = self.predict(X)
        self.trees[j].append(
            LocalBoostingTree(
                distribution=self.distribution,
                max_depth=self.max_depth[j],
                min_samples_split=self.min_samples_split[j],
                min_samples_leaf=self.min_samples_leaf[j],
            )
        )
        self.trees[j][-1].fit_gradients(X=X, y=y, z=z, j=j)
        self.n_estimators[j] += 1

    def predict_parameter(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the parameter values for the input data.

        :param X: Input data matrix of shape (n, p).
        :return: Predicted parameter values for the input data of shape (n, p).
        """
        return np.tile(self.beta0, (X.shape[0], 1)).T + np.array(
            [
                sum(
                    [
                        self.learning_rate[j] * self.trees[j][k].predict(X)
                        for k in range(self.n_estimators[j])
                    ]
                )
                if self.n_estimators[j] > 0
                else np.zeros(X.shape[0])
                for j in range(self.p)
            ]
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the response for the input data.

        :param X: Input data matrix of shape (n, p).
        :return: Predicted response values for the input data of shape (n,).
        """
        X, _ = fix_datatype(X=X, feature_names=self.feature_names)
        beta = self.predict_parameter(X=X)
        return self.z0 + np.sum(beta.T * X, axis=1)

    def compute_feature_importances(
        self, j: Union[str, int] = "all", normalize: bool = True
    ) -> np.ndarray:
        """
        Computes the feature importance for parameter all features for dimension j
        If j is 'all', the feature importance is calculated over all parameter dimensions.
        Note that the feature importance is calculated for regression attentions beta_j(x) meaning that the GLM parameters are not taken into account.

        :param j: Parameter dimension. If 'all', calculate importance over all parameter dimensions.
        :return: Feature importance of shape (n_features,)
        """
        if j == "all":
            feature_importances = (
                (
                    np.array(
                        [
                            [
                                tree.compute_feature_importances()
                                for tree in self.trees[j]
                            ]
                            for j in range(self.p)
                        ]
                    ).sum(axis=(0, 1))
                )
                if len(self.trees[0]) > 0
                else np.zeros(self.p)
            )
        else:
            feature_importances = (
                np.array(
                    [tree.compute_feature_importances() for tree in self.trees[j]]
                ).sum(axis=0)
                if len(self.trees[j]) > 0
                else np.zeros(self.p)
            )
        if normalize and sum(feature_importances) > 0:
            feature_importances /= feature_importances.sum()

        return feature_importances
