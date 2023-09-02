from typing import Tuple, Union, Optional, List
import warnings

import numpy as np
import pandas as pd


def fix_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
    w: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
    feature_names: Optional[List[str]] = None,
    parallel_fit: Optional[List[List[int]]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert data to numpy arrays if they are pandas dataframes or series.
    Also make sure feature names and parallel order is correct.

    :param X: Input data matrix of shape (n_samples, n_features).
    :param y: True response values for the input data.
    :param w: Weights of the observations.
    :param feature_names: Names of the features in X.
    :param parallel_fit: Lists of features to fit in parallel. If None, no features are fit in parallel.
    """
    if isinstance(X, pd.DataFrame):
        if feature_names is not None:
            X = X[feature_names]
        X = X.to_numpy()
    if y is None:
        return X
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    if isinstance(w, (pd.Series, pd.DataFrame)):
        w = w.to_numpy()
    if parallel_fit is not None:
        parallel_features = [
            feature for feature_list in parallel_fit for feature in feature_list
        ]

        cyclical_features = [j for j in range(X.shape[1]) if j not in parallel_features]
        X_ordered = X[:, cyclical_features + parallel_features]
        if np.any(X != X_ordered):
            X = X_ordered
            warnings.warn(
                "X was not ordered correctly. Reordering X to match parallel_fit order."
            )
    return X, y, w
