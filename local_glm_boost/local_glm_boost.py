from typing import TypeVar, List, Union, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .utils.distributions import initiate_distribution
from .utils.fix_datatype import fix_datatype
from .boosting_tree import BoostingTree


T = TypeVar("T")
Hyperparameter = Union[T, List[T], Dict[Union[int, str], T]]
FeatureDict = Union[
    Dict[Union[int, str], List[Union[int, str]]],
    Dict[Union[int, str], Dict[Union[int, str], List[Union[int, str]]]],
]


class LocalGLMBooster:
    def __init__(
        self,
        distribution: str = "normal",
        n_estimators: Hyperparameter[int] = 100,
        learning_rate: Hyperparameter[float] = 0.1,
        min_samples_split: Hyperparameter[int] = 2,
        min_samples_leaf: Hyperparameter[int] = 1,
        max_depth: Hyperparameter[int] = 3,
        glm_init: Hyperparameter[bool] = True,
        feature_selection: Optional[FeatureDict] = None,
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
        :param feature_selection: Features to use for each coefficient. A dictionary with the coefficient name or number as key and a list of feature names or numbers as value.
        """
        self.distribution = initiate_distribution(distribution)

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.glm_init = glm_init
        self.feature_selection = feature_selection

        self.p = None
        self.beta0 = None
        self.z0 = None
        self.trees = None
        self.feature_names = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
    ):
        """
        Fit the model to the data.

        :param X: Input data matrix of shape (n, p).
        :param y: True response values for the input data of shape (n,).
        :param w: Weights of the observations. If `None`, all weights are set to 1.
        """
        self.feature_names = self._get_feature_names(X=X)
        X, y, w = fix_datatype(X=X, y=y, w=w if w is not None else np.ones_like(y))
        self.p = X.shape[1]
        self._adjust_hyperparameters()
        self.z0, self.beta0 = self._adjust_initializer(X=X, y=y, w=w)
        self._initiate_trees()

        z = self.z0 + (self.beta0.T @ X.T).T.reshape(-1)

        for k in range(max(self.n_estimators.values())):
            for j in range(self.p):
                if k < self.n_estimators[j]:
                    self.trees[j][k].fit_gradients(
                        X=X, y=y, z=z, w=w, j=j, features=self.feature_selection[j]
                    )
                    z += (
                        self.learning_rate[j]
                        * self.trees[j][k].predict(X[:, self.feature_selection[j]])
                        * X[:, j]
                    )

    def _get_feature_names(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> List[Union[int, str]]:
        """Get the feature names from the input data.
        If the input data is a DataFrame, the column names are returned.
        Otherwise, the features are named 0, 1, ..., p-1.
        """
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        else:
            return list(np.arange(X.shape[1]))

    def _adjust_hyperparameters(self) -> None:
        """Adjust hyperparameters given the covariate dimensions.
        Since the accepted format of the hyperparameters are floats, lists, or Series, they are here adjusted
        to lists of length p. The covariate is adjusted to comply with the data in case of a Series.
        """

        # TODO: Replace with a type check and check if it is a Hyperparameter?
        for parameter_name in [
            "n_estimators",
            "learning_rate",
            "min_samples_split",
            "min_samples_leaf",
            "max_depth",
            "glm_init",
            "feature_selection",
        ]:
            parameter_value = getattr(self, parameter_name)
            if isinstance(parameter_value, Dict):
                new_hyperparam_dict = {}
                for j, feature_name in enumerate(self.feature_names):
                    if feature_name not in parameter_value.keys():
                        raise ValueError(
                            f"Hyperparameter {parameter_name} missing for feature {feature_name}"
                        )
                    # Adjust feature names for feature selection
                    if parameter_name == "feature_selection":
                        new_hyperparam_dict[j] = [
                            self.feature_names.index(feature)
                            for feature in parameter_value[feature_name]
                        ]
                    else:
                        new_hyperparam_dict[j] = parameter_value[feature_name]
                    setattr(self, parameter_name, new_hyperparam_dict)
            elif isinstance(parameter_value, List):
                if len(parameter_value) != self.p:
                    raise ValueError(
                        f"Hyperparameter {parameter_name} not of length {self.p}"
                    )
                setattr(
                    self, parameter_name, {j: parameter_value[j] for j in range(self.p)}
                )
            elif parameter_name == "feature_selection" and parameter_value is None:
                setattr(
                    self,
                    parameter_name,
                    {j: [k for k in range(self.p)] for j in range(self.p)},
                )
            else:
                setattr(
                    self, parameter_name, {j: parameter_value for j in range(self.p)}
                )

    def _initiate_trees(self):
        """Initiate the trees."""
        self.trees = [
            [
                BoostingTree(
                    distribution=self.distribution,
                    max_depth=self.max_depth[j],
                    min_samples_split=self.min_samples_split[j],
                    min_samples_leaf=self.min_samples_leaf[j],
                )
                for _ in range(self.n_estimators[j])
            ]
            for j in range(self.p)
        ]

    def _adjust_initializer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Adjust the initalization of the model.
        This will be done as a GLM for all dimensions specified in the model attribute glm_init.
        If all glm_init are False, the initialization is a constant MLE since the intercept is still estimated.

        :param X: The input training data for the model as a numpy array.
        :param y: The target values.
        :param z: The current parameter estimates. If None, the initial parameter estimates are assumed to be zero.
        :param w: The weights of the observations. If None, all weights are set to 1.
        :return: The initial parameter estimates.
        """
        if w is None:
            w = np.ones_like(y)
        if z is None:
            z = np.zeros(X.shape[0])
        features_to_initiate = [glm_init for glm_init in self.glm_init.values()]
        to_minimize = lambda z0_and_beta: self.distribution.loss(
            y=y,
            z=z + z0_and_beta[0] + X[:, features_to_initiate] @ z0_and_beta[1:],
            w=w,
        ).sum()
        glm_coefficients = minimize(
            fun=to_minimize,
            x0=np.zeros(1 + sum(features_to_initiate)),
        )["x"]
        z0 = glm_coefficients[0]
        beta0 = np.zeros(X.shape[1])
        beta0[features_to_initiate] = glm_coefficients[1:]
        return z0, beta0

    def add_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: int,
        z: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
    ) -> None:
        """
        Updates the current boosting model with one additional tree to the jth coefficient.

        :param X: The training input data, shape (n_samples, n_features).
        :param y: The target values for the training data.
        :param j: Coefficient  to update
        :param z: The current predictions of the model. If None, the predictions are computed from the current model.
        :param w: The weights of the observations. If None, all weights are set to 1.
        """
        if w is None:
            w = np.ones_like(y)
        X, y, w = fix_datatype(X=X, y=y, w=w)
        if z is None:
            z = self.predict(X)
        self.trees[j].append(
            BoostingTree(
                distribution=self.distribution,
                max_depth=self.max_depth[j],
                min_samples_split=self.min_samples_split[j],
                min_samples_leaf=self.min_samples_leaf[j],
            )
        )
        self.trees[j][-1].fit_gradients(
            X=X, y=y, z=z, w=w, j=j, features=self.feature_selection[j]
        )
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
                        * self.trees[j][k].predict(X[:, self.feature_selection[j]])
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
        X_fixed = fix_datatype(X=X, feature_names=self.feature_names)
        beta = self.predict_parameter(X=X_fixed)
        z_hat = self.z0 + np.sum(beta.T * X_fixed, axis=1)
        if isinstance(X, pd.DataFrame):
            return pd.Series(z_hat, index=X.index)
        else:
            return z_hat

    def compute_feature_importances(
        self, j: Union[str, int] = "all", normalize: bool = True
    ) -> np.ndarray:
        """
        Computes the feature importance for parameter all feature_selection for dimension j
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
                feature_importances[
                    self.feature_selection[j]
                ] += feature_importances_from_trees
        else:
            if isinstance(j, str):
                j = self.feature_names.index(j)

            feature_importances = np.zeros(self.p)
            feature_importances_from_trees = np.array(
                [tree.compute_feature_importances() for tree in self.trees[j]]
            ).sum(axis=0)

            feature_importances[
                self.feature_selection[j]
            ] = feature_importances_from_trees
        if normalize:
            feature_importances /= feature_importances.sum()

        if self.feature_names is not None:
            feature_importances = pd.Series(
                feature_importances, index=self.feature_names
            )
        return feature_importances

    def reset(self, n_estimators: Optional[Union[int, List[int]]] = None) -> None:
        """
        Resets the model to the initial state.
        If n_estimators is not None, the number of trees in each dimension is reset to the specified value.

        :param n_estimators: Number of trees in each dimension.
        """
        if n_estimators is not None:
            self.n_estimators = n_estimators

        self.trees = None
        self.z0 = None
        self.beta0 = None
        self.feature_names = None
