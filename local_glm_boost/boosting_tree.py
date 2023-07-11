from typing import List

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize

from local_glm_boost.utils.distributions import Distribution


class BoostingTree(DecisionTreeRegressor):
    """
    A Gradient Boosting Machine tree for the LocalGLMBoost algorithm.

    :param max_depth: The maximum depth of the tree.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
    """

    def __init__(
        self,
        distribution: Distribution,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
    ):
        """
        Constructs a new GBMTree instance.

        :param distribution: The distribution of the response variable.
        :param max_depth: The maximum depth of the tree.
        :param min_samples_split: The minimum number of samples required to split an internal node.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.distribution = distribution

    def fit_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        j: int,
        features: List[int],
    ) -> None:
        """
        Fits the BoostingTree to the negative gradients and adjusts node values to minimize loss.

        :param X: The training input samples.
        :param y: The target values.
        :param z: The predicted parameter values from the previous iteration.
        :param j: The index of the current iteration.
        :param features: The indices of the features to use for the tree.
        """
        g = X[:, j] * self.distribution.grad(y=y, z=z)
        self.fit(X[:, features], -g)
        self._adjust_node_values(X=X, y=y, z=z, j=j)

    def _adjust_node_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        j: int,
        node_index: int = 0,
    ) -> None:
        """
        Adjust the predicted node values of the node outputs to its optimal step size.
        Adjustment is performed recursively starting at the top of the tree.

        :param X: The input training data for the model as a numpy array
        :param y: The output training data for the model as a numpy array
        :param z: The current parameter estimates
        :param j: Parameter dimension to update
        :param node_index: The index of the node to update
        """
        # Optimize node response
        g_opt = minimize(
            fun=lambda step: self.distribution.loss(y=y, z=z + X[:, j] * step).sum(),
            x0=self.tree_.value[node_index][0],
        ).x[0]
        self.tree_.value[node_index] = g_opt
        self.tree_.impurity[node_index] = self.distribution.loss(
            y=y,
            z=z + X[:, j] * g_opt,
        ).sum()

        # Tend to the children
        feature = self.tree_.feature[node_index]
        if feature == -2:
            # This is a leaf
            return
        threshold = self.tree_.threshold[node_index]
        index_left = X[:, feature] <= threshold
        child_left = self.tree_.children_left[node_index]
        child_right = self.tree_.children_right[node_index]
        self._adjust_node_values(
            X=X[index_left],
            y=y[index_left],
            z=z[index_left],
            j=j,
            node_index=child_left,
        )
        self._adjust_node_values(
            X=X[~index_left],
            y=y[~index_left],
            z=z[~index_left],
            j=j,
            node_index=child_right,
        )

    def compute_feature_importances(self) -> np.ndarray:
        """
        Returns the feature importances of the tree.

        :return: The feature importances of the tree.
        """
        return self.tree_.compute_feature_importances(normalize=False)
