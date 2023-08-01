from typing import List, Optional

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
        self.tree_ = None

    def fit_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        j: int,
        features: List[int],
        w: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fits the BoostingTree to the negative gradients and adjusts node values to minimize loss.

        :param X: The training input samples.
        :param y: The target values.
        :param z: The predicted parameter values from the previous iteration.
        :param j: The index of the current iteration.
        :param features: The indices of the features to use for the tree.
        :param w: The weights of the observations. If `None`, all weights are set to 1.
        """
        if w is None:
            w = np.ones_like(y)
        g = X[:, j] * self.distribution.grad(y=y, z=z, w=w)
        self.fit(X[:, features], -g)
        self._adjust_node_values(X=X, y=y, z=z, w=w, j=j, features=features)

    def _adjust_node_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        j: int,
        features: List[int],
        node_index: int = 0,
    ) -> None:
        """
        Adjust the predicted node values of the node outputs to its optimal step size.
        Adjustment is performed recursively starting at the top of the tree.

        :param X: The input training data for the model as a numpy array
        :param y: The output training data for the model as a numpy array
        :param z: The current parameter estimates
        :param w: The weights of the observations.
        :param j: Parameter dimension to update
        :param features: The indices of the features to use for the tree.
        :param node_index: The index of the node to update
        """
        node_loss = lambda step: self.distribution.loss(
            y=y, z=z + X[:, j] * step, w=w
        ).sum()
        node_value = minimize(
            fun=node_loss,
            x0=self.tree_.value[node_index][0],
        ).x[0]
        self.tree_.value[node_index] = node_value
        self.tree_.impurity[node_index] = node_loss(node_value)

        # Tend to the children
        feature = self.tree_.feature[node_index]
        if feature == -2:
            # This is a leaf
            return
        threshold = self.tree_.threshold[node_index]
        index_left = X[:, features[feature]] <= threshold
        child_left = self.tree_.children_left[node_index]
        child_right = self.tree_.children_right[node_index]
        self._adjust_node_values(
            X=X[index_left],
            y=y[index_left],
            z=z[index_left],
            w=w[index_left],
            j=j,
            features=features,
            node_index=child_left,
        )
        self._adjust_node_values(
            X=X[~index_left],
            y=y[~index_left],
            z=z[~index_left],
            w=w[~index_left],
            j=j,
            features=features,
            node_index=child_right,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the predicted values of the tree.

        :param X: The input samples.
        :return: The predicted values of the tree.
        """
        if self.tree_ is None:
            return np.zeros(X.shape[0])
        else:
            return super().predict(X)

    def compute_feature_importances(self) -> np.ndarray:
        """
        Returns the feature importances of the tree.

        :return: The feature importances of the tree.
        """
        if self.tree_ is None:
            return np.zeros(0)
        else:
            return self.tree_.compute_feature_importances(normalize=False)
