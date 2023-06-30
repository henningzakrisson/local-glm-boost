from typing import List, Optional, Union

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class LocalGLMBooster:
    def __init__(
        self,
        kappa: Union[int, List[int]] = 100,
        eps: Union[float, List[float]] = 0.1,
        max_depth: Union[int, List[int]] = 2,
        min_samples_leaf: Union[int, List[int]] = 20,
    ):
        """
        :param kappa: Number of boosting steps. Dimension-wise or global for all parameter dimensions.
        :param eps: Shrinkage factors, which scales the contribution of each tree. Dimension-wise or global for all parameter dimensions.
        :param max_depth: Maximum depths of each decision tree. Dimension-wise or global for all parameter dimensions.
        :param min_samples_leaf: Minimum number of samples required at a leaf node. Dimension-wise or global for all parameter dimensions.
        """
        self.kappa = kappa
        self.eps = eps
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.p = None
        self.z0 = None
        self.beta0 = None
        self.trees = None

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
        self.p = X.shape[1]
        self.z0 = y.mean()
        self.trees = [[None] * self.kappa for j in range(self.p)]
        beta = np.zeros((self.p, X.shape[0]))
        z = self.z0 + np.sum(beta.T * X, axis=1)

        for k in range(self.kappa):
            for j in range(self.p):
                g = -2 * X[:, j] * (y - z)
                self.trees[j][k] = GradientBoostingRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    n_estimators=1,
                    learning_rate=self.eps,
                )
                self.trees[j][k].fit(X, -g)
                beta_add = self.trees[j][k].predict(X)
                beta[j] += beta_add
                z += beta_add * X[:, j]

    def predict_parameter(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the parameter values for the input data.

        :param X: Input data matrix of shape (n, p).
        :return: Predicted parameter values for the input data of shape (n, p).
        """
        beta = np.zeros((self.p, X.shape[0]))
        for j in range(self.p):
            for k in range(self.kappa):
                beta[j] += self.trees[j][k].predict(X)
        return beta

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


if __name__ == "__main__":
    n = 10000
    p = 1
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))

    mu = 1 + X[:, 0] * (X[:, 0] > 0)
    y = rng.normal(mu, 1)

    model = LocalGLMBooster(
        kappa=100,
        eps=0.01,
        max_depth=2,
        min_samples_leaf=20,
    )
    model.fit(X, y)

    print(f"Intercept loss: {np.sum((y-y.mean())**2)}")
    print(f"Model loss: {np.sum((y-model.predict(X))**2)}")
