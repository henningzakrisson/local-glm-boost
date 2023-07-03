from typing import Union, List, Dict, Tuple, Optional
import warnings

import numpy as np

from local_glm_boost import LocalGLMBooster
from distributions import initiate_distribution, Distribution


def _fold_split(
    X: np.ndarray,
    n_splits: int = 4,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split data into k folds.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param random_state: The seed used by the random number generator.
    :param rng: The random number generator.
    :return List of tuples containing (idx_train, idx_test) for each fold.
    """
    if rng is None:
        rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    idx = rng.permutation(n_samples)
    idx_folds = np.array_split(idx, n_splits)
    folds = []
    for i in range(n_splits):
        idx_test = idx_folds[i]
        idx_train = np.concatenate(idx_folds[:i] + idx_folds[i + 1 :])
        folds.append((idx_train, idx_test))
    return folds


def tune_kappa(
    X: np.ndarray,
    y: np.ndarray,
    kappa_max: Union[int, List[int]] = 1000,
    eps: Union[float, List[float]] = 0.1,
    max_depth: Union[int, List[int]] = 2,
    min_samples_leaf: Union[int, List[int]] = 20,
    distribution: Union[str, Distribution] = "normal",
    n_splits: int = 4,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Union[List[int], np.ndarray]]:
    """Tunes the kappa parameter of a CycGBM model using k-fold cross-validation.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target vector of shape (n_samples,).
    :param kappa_max: The maximum value of the kappa parameter to test. Dimension-wise or same for all parameter dimensions.
    :param eps: The epsilon parameters for the CycGBM model.Dimension-wise or same for all parameter dimensions.
    :param max_depth: The maximum depth of the decision trees in the GBM model. Dimension-wise or same for all parameter dimensions.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node in the CycGBM model. Dimension-wise or same for all parameter dimensions.
    :param distribution: The distribution of the target variable.
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param random_state: The random state to use for the k-fold split.
    :param rng: The random number generator.
    :return: A dictionary containing the following keys:
        - "kappa": The optimal kappa parameter value for each parameter dimension.
        - "loss": The loss values for each kappa parameter value.
    """
    if rng is None:
        rng = np.random.default_rng(random_state)
    folds = _fold_split(X=X, n_splits=n_splits, rng=rng)
    if isinstance(distribution, str):
        distribution = initiate_distribution(distribution=distribution)
    p = X.shape[1]
    kappa_max = kappa_max if isinstance(kappa_max, list) else [kappa_max] * p
    loss = np.ones((n_splits, max(kappa_max) + 1, p)) * np.nan
    for i, idx in enumerate(folds):
        idx_train, idx_valid = idx
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        model = LocalGLMBooster(
            kappa=0,
            eps=eps,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            distribution=distribution,
        )
        model.fit(X_train, y_train)
        z_valid = model.predict(X_valid)
        loss[i, 0, :] = model.distribution.loss(y=y_valid, z=z_valid).sum()

        for k in range(1, max(kappa_max) + 1):
            for j in range(p):
                if k < kappa_max[j]:
                    model.update(X=X_train, y=y_train, j=j)
                    beta_add = model.eps[j] * model.trees[j][-1].predict(X_valid)
                    z_valid += beta_add * X_valid[:, j]
                    loss[i, k, j] = model.distribution.loss(y=y_valid, z=z_valid).sum()
                else:
                    if j == 0:
                        loss[i, k, j] = loss[i, k - 1, j + 1]
                    else:
                        loss[i, k, j] = loss[i, k, j - 1]

            # Stop if no improvement was made
            if k != max(kappa_max) and np.all(
                [loss[i, k, 0] >= loss[i, k - 1, 1]]
                + [loss[i, k, j] >= loss[i, k, j - 1] for j in range(1, p)]
            ):
                loss[i, k + 1 :, :] = loss[i, k, -1]
                break

            if k == max(kappa_max):
                warnings.warn(
                    "Maximum kappa value was reached without convergence. "
                    "Consider increasing the maximum kappa value."
                )

    loss_total = loss.sum(axis=0)
    loss_delta = np.zeros((p, max(kappa_max) + 1))
    loss_delta[0, 1:] = loss_total[1:, 0] - loss_total[:-1, -1]
    for j in range(1, p):
        loss_delta[j, 1:] = loss_total[1:, j] - loss_total[1:, j - 1]
    kappa = np.maximum(0, np.argmax(loss_delta > 0, axis=1) - 1)
    did_not_converge = (loss_delta > 0).sum(axis=1) == 0
    for j in range(p):
        if did_not_converge[j] and kappa_max[j] > 0:
            warnings.warn(f"tuning did not converge for dimension {j}")
            kappa[j] = kappa_max[j]

    return {"kappa": kappa, "loss": loss}


if __name__ == "__main__":
    n = 10000
    p = 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    z0 = 0
    beta0 = np.sin(5 * X[:, 1])
    beta1 = X[:, 0]
    beta = np.stack([beta0, beta1], axis=1).T

    mu = z0 + np.sum(beta.T * X, axis=1)
    y = rng.normal(mu, 0.1)

    kappa_max = 300
    eps = [0.1, 0.01]
    max_depth = 2
    min_samples_leaf = 20
    distribution = "normal"
    n_splits = 4
    random_state = 0

    tuning_results = tune_kappa(
        X=X,
        y=y,
        kappa_max=kappa_max,
        eps=eps,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        distribution=distribution,
        n_splits=n_splits,
        random_state=random_state,
    )

    for j in range(p):
        print(f"optimal kappa for coefficient {j}: {tuning_results['kappa'][j]}")
