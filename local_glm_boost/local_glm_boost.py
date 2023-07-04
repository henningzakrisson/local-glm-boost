from typing import List, Union, Optional
import warnings

import numpy as np

from distributions import Distribution, initiate_distribution
from local_boosting_tree import LocalBoostingTree


class LocalGLMBooster:
    def __init__(
        self,
        p: Optional[int] = 1,
        kappa: Union[List[int], int] = 100,
        eps: Union[List[float], float] = 0.1,
        max_depth: Union[List[int], int] = 2,
        min_samples_leaf: Union[List[int], int] = 20,
        distribution: Union[Distribution, str] = "normal",
    ):
        """
        :param kappa: Number of boosting steps. Dimension-wise or global for all coefficients.
        :param eps: Shrinkage factors, which scales the contribution of each tree. Dimension-wise or global for all coefficients
        :param max_depth: Maximum depths of each decision tree. Dimension-wise or global for all coefficients.
        :param min_samples_leaf: Minimum number of samples required at a leaf node. Dimension-wise or global for all coefficients.
        :param distribution: The distribution of the response variable. A Distribution object or a string.
        """
        self.kappa = kappa
        self.eps = eps
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        if isinstance(distribution, str):
            self.distribution = initiate_distribution(distribution)
        else:
            self.distribution = distribution

        self.p = None
        self.beta0 = None
        self.z0 = None
        self.trees = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        glm_initialization: bool = True,
    ):
        """
        Fit the model to the data.

        :param X: Input data matrix of shape (n, p).
        :param y: True response values for the input data of shape (n,).
        """
        self.p = X.shape[1]
        self._adjust_hyperparameters()
        self.trees = [
            [
                LocalBoostingTree(
                    max_depth=self.max_depth[j],
                    min_samples_leaf=self.min_samples_leaf[j],
                    distribution=self.distribution,
                )
                for _ in range(self.kappa[j])
            ]
            for j in range(self.p)
        ]

        self.z0 = self.distribution.mle(y=y)
        if glm_initialization:
            self.beta0 = self.distribution.glm_initialization(X=X, y=y, z0=self.z0)
        else:
            self.beta0 = np.zeros((self.p, 1))

        beta = np.tile(self.beta0, X.shape[0])
        z = self.z0 + np.sum(beta.T * X, axis=1)

        for k in range(max(self.kappa)):
            for j in range(self.p):
                if k < self.kappa[j]:
                    self.trees[j][k].fit_gradients(X=X, y=y, z=z, j=j)
                    beta_add = self.trees[j][k].predict(X)
                    beta[j] += self.eps[j] * beta_add
                    z += self.eps[j] * beta_add * X[:, j]

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: int,
    ) -> None:
        """
        Updates the current boosting model with one additional tree

        :param X: The training input data, shape (n_samples, n_features).
        :param y: The target values for the training data.
        :param j: Coefficient  to update
        """
        z = self.predict(X)
        self.trees[j].append(
            LocalBoostingTree(
                max_depth=self.max_depth[j],
                min_samples_leaf=self.min_samples_leaf[j],
                distribution=self.distribution,
            )
        )
        self.trees[j][-1].fit_gradients(X=X, y=y, z=z, j=j)
        self.kappa[j] += 1

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

        for param in ["kappa", "eps", "max_depth", "min_samples_leaf"]:
            adjust_param(param)

    def predict_parameter(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the parameter values for the input data.

        :param X: Input data matrix of shape (n, p).
        :return: Predicted parameter values for the input data of shape (n, p).
        """
        return np.tile(self.beta0, X.shape[0]) + np.array(
            [
                sum(
                    [
                        self.eps[j] * self.trees[j][k].predict(X)
                        for k in range(self.kappa[j])
                    ]
                )
                if self.kappa[j] > 0
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
        beta = self.predict_parameter(X=X)
        return self.z0 + np.sum(beta.T * X, axis=1)

    def feature_importances(
        self, j: Union[str, int] = "all", normalize: bool = True
    ) -> np.ndarray:
        """
        Computes the feature importances for parameter dimension j
        Note that the GLM initialization is NOT taken into consideration when computing feature importances.

        :param j: Parameter dimension. If 'all', calculate importance over all parameter dimensions.
        :return: Feature importance of shape (n_features,)
        """
        if j == "all":
            feature_importances = (
                (
                    np.array(
                        [
                            [tree.feature_importances() for tree in self.trees[j]]
                            for j in range(self.p)
                        ]
                    ).sum(axis=(0, 1))
                )
                if len(self.trees[0]) > 0
                else np.zeros(self.p)
            )
        else:
            feature_importances = (
                np.array([tree.feature_importances() for tree in self.trees[j]]).sum(
                    axis=0
                )
                if len(self.trees[j]) > 0
                else np.zeros(self.p)
            )
        if normalize:
            feature_importances /= feature_importances.sum()

        if np.any(self.beta0[j] != 0):
            warnings.warn(
                "The feature importances do not take the GLM initialization into account."
            )
        return feature_importances


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from tune_kappa import tune_kappa
    from logger import LocalGLMBoostLogger

    n = 200000
    p = 8
    rng = np.random.default_rng(0)
    cov = np.eye(p)
    cov[1, 7] = cov[7, 1] = 0.5
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    z0 = 0

    beta0 = 0.5 * np.ones(n)
    beta1 = -0.5 * X[:, 1]
    beta2 = np.abs(X[:, 2]) * np.sin(2 * X[:, 2]) / X[:, 2]
    beta3 = 0.5 * X[:, 4]
    beta4 = (1 / 8) * X[:, 5] ** 2
    beta5 = np.zeros(n)
    beta6 = np.zeros(n)
    beta7 = np.zeros(n)
    beta = np.stack([beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7], axis=1).T

    mu = z0 + np.sum(beta.T * X, axis=1)
    y = rng.normal(mu, 1)

    y_train, y_test, X_train, X_test = train_test_split(
        y, X, test_size=0.5, random_state=1
    )

    max_depth = 2
    min_samples_leaf = 10
    distribution = "normal"
    kappa_max = 1000
    eps = 0.01

    logger = LocalGLMBoostLogger(verbose=2)

    tuning_results = tune_kappa(
        X=X_train,
        y=y_train,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        distribution=distribution,
        kappa_max=kappa_max,
        eps=eps,
        n_splits=2,
        random_state=2,
        logger=logger,
    )

    kappa_opt = tuning_results["kappa"]

    for j in range(p):
        print(f"Optimal kappa for covariate {j}: {kappa_opt[j]}")

    model = LocalGLMBooster(
        kappa=kappa_opt,
        eps=eps,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        distribution="normal",
    )
    model.fit(X, y, glm_initialization=True)

    print(f"Intercept MSE: {np.mean((y_test-y_train.mean())**2)}")
    print(f"GLM MSE: {np.mean((y_test-model.z0 - model.beta0.T @ X_test.T)**2)}")
    print(f"Model MSE: {np.mean((y_test-model.predict(X_test))**2)}")

    for j in range(p):
        feature_importances = model.feature_importances(j=j, normalize=True)
        for k in range(p):
            print(
                f"Feature importance for covariate {j} on beta_{k}: {feature_importances[k]}"
            )
