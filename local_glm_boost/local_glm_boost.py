from typing import List, Union, Optional
import warnings

import numpy as np

from .distributions import Distribution, initiate_distribution
from .local_boosting_tree import LocalBoostingTree


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
        glm_init: bool = True,
    ):
        """
        Fit the model to the data.

        :param X: Input data matrix of shape (n, p).
        :param y: True response values for the input data of shape (n,).
        :param glm_init: Whether to initialize the model with a GLM fit.
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

        if glm_init:
            self.z0, self.beta0 = self.distribution.glm(X=X, y=y)
        else:
            self.z0 = self.distribution.mle(y=y)
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

        # Re-adjust the initial parameter values
        if glm_init:
            self.z0, self.beta0 = self.distribution.glm(X=X, y=y, z=z)
        else:
            self.z0 = self.distribution.mle(y=y, z=z)

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
        Note that the feature importance is calculated for regression attentions beta_j(x) meaning that the GLM parameters are not taken into account.

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
        if normalize and sum(feature_importances) > 0:
            feature_importances /= feature_importances.sum()

        return feature_importances


if __name__ == "__main__":
    from .tune_kappa import tune_kappa
    from .logger import LocalGLMBoostLogger

    n = 20000
    p = 3
    rng = np.random.default_rng(0)
    cov = np.eye(p)
    # cov[1, 7] = cov[7, 1] = 0.5
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    z0 = 0

    betas = [[]] * p
    betas[0] = 0.5 * np.ones(n)
    betas[1] = -0.5 * X[:, 1]
    betas[2] = np.sin(2 * X[:, 0])
    # betas[3] = 0.5 * X[:, 4]
    # betas[4] = (1 / 8) * X[:, 5] ** 2
    # betas[5] = np.zeros(n)
    # betas[6] = np.zeros(n)
    # betas[7] = np.zeros(n)
    beta = np.stack(betas, axis=1).T

    mu = z0 + np.sum(beta.T * X, axis=1)
    y = rng.normal(mu, 1)

    idx = np.arange(n)
    rng.shuffle(idx)
    idx_train, idx_test = idx[: int(0.5 * n)], idx[int(0.5 * n) :]
    X_train, y_train, mu_train = X[idx_train], y[idx_train], mu[idx_train]
    X_test, y_test, mu_test = X[idx_test], y[idx_test], mu[idx_test]

    max_depth = 2
    min_samples_leaf = 10
    distribution = "normal"
    kappa_max = 100
    eps = 0.1

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
    model.fit(X, y, glm_init=True)

    print(f"True MSE: {np.mean((y_test-mu_test)**2)}")
    print(f"Intercept MSE: {np.mean((y_test-y_train.mean())**2)}")
    print(f"GLM MSE: {np.mean((y_test-model.z0 - model.beta0.T @ X_test.T)**2)}")
    print(f"Model MSE: {np.mean((y_test-model.predict(X_test))**2)}")

    feature_importances = [
        model.feature_importances(j=j, normalize=True) for j in range(p)
    ]
    for j in range(p):
        for k in range(p):
            print(
                f"Feature importance for covariate {k} on beta_{j}: {feature_importances[k][j]}"
            )
