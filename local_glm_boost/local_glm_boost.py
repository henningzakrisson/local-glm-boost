from typing import List, Union, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
        self.features = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    ):
        """
        Fit the model to the data.

        :param X: Input data matrix of shape (n, p).
        :param y: True response values for the input data of shape (n,).
        :param features: A dictionary of features to be used for each covariate. If None, all covariates are used.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            if features is not None:
                self.features = {
                    X.columns.get_loc(coefficient): [
                        X.columns.get_loc(feature) for feature in features[coefficient]
                    ]
                    for coefficient in X.columns
                }
        X, y = fix_datatype(X=X, y=y)
        self.p = X.shape[1]
        if features is None:
            self.features = {j: list(range(X.shape[1])) for j in range(self.p)}
        elif self.features is None:
            self.features = features

        self._adjust_hyperparameters()
        self.z0, self.beta0 = self.adjust_initializer(X=X, y=y)
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

        z = self.z0 + (self.beta0.T @ X.T).T.reshape(-1)

        for k in range(max(self.n_estimators)):
            for j in range(self.p):
                if k < self.n_estimators[j]:
                    self.trees[j][k].fit_gradients(
                        X=X, y=y, z=z, j=j, features=self.features[j]
                    )
                    z += (
                        self.learning_rate[j]
                        * self.trees[j][k].predict(X[:, self.features[j]])
                        * X[:, j]
                    )

        # Re-adjust the initial parameter values
        self.z0, self.beta0 = self.adjust_initializer(X=X, y=y)

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

    def adjust_initializer(
        self, X: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Adjust the initalization of the model.
        This will be done as a GLM for all dimensions specified in the model attribute glm_init.
        If all glm_init are False, the initialization is a constant MLE since the intercept is still estimated.

        :param X: The input training data for the model as a numpy array.
        :param y: The target values.
        :param z: The current parameter estimates. If None, the initial parameter estimates are assumed to be zero.
        :return: The initial parameter estimates.
        """
        if z is None:
            z = np.zeros(X.shape[0])
        glm_coefficients = minimize(
            fun=lambda beta: self.distribution.loss(
                y=y, z=z + beta[0] + X[:, self.glm_init] @ beta[1:]
            ).sum(),
            x0=np.zeros(1 + sum(self.glm_init)),
        )["x"]
        z0 = glm_coefficients[0]
        beta0 = np.zeros(X.shape[1])
        beta0[self.glm_init] = glm_coefficients[1:]
        return z0, beta0

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
        self.trees[j][-1].fit_gradients(X=X, y=y, z=z, j=j, features=self.features[j])
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
                        self.learning_rate[j]
                        * self.trees[j][k].predict(X[:, self.features[j]])
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
            feature_importances = np.zeros(self.p)
            for j in range(self.p):
                feature_importances_from_trees = np.array(
                    [tree.compute_feature_importances() for tree in self.trees[j]]
                ).sum(axis=0)
                feature_importances[self.features[j]] += feature_importances_from_trees
        else:
            if isinstance(j, str):
                j = self.feature_names.get_loc(j)

            feature_importances = np.zeros(self.p)
            feature_importances_from_trees = np.array(
                [tree.compute_feature_importances() for tree in self.trees[j]]
            ).sum(axis=0)

            feature_importances[self.features[j]] = feature_importances_from_trees
        if normalize:
            feature_importances /= feature_importances.sum()

        if self.feature_names is not None:
            feature_importances = pd.Series(
                feature_importances, index=self.feature_names
            )
        return feature_importances
