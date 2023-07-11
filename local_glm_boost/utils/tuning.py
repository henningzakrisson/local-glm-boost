from typing import Union, List, Dict, Tuple, Optional

import numpy as np

from local_glm_boost.local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.distributions import initiate_distribution, Distribution
from .logger import LocalGLMBoostLogger


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


def tune_n_estimators(
    X: np.ndarray,
    y: np.ndarray,
    distribution: Union[str, Distribution] = "normal",
    learning_rate: Union[float, List[float]] = 0.1,
    n_estimators_max: Union[int, List[int]] = 1000,
    min_samples_split: Union[int, List[int]] = 2,
    min_samples_leaf: Union[int, List[int]] = 20,
    max_depth: Union[int, List[int]] = 2,
    features: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    glm_init: Union[List[bool], bool] = True,
    n_splits: int = 4,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    logger: Optional[LocalGLMBoostLogger] = None,
) -> Dict[str, Union[List[int], Dict[str, np.ndarray]]]:
    """Tunes the kappa parameter of a CycGBM model using k-fold cross-validation.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target values of shape (n_samples,).
    :param distribution: The distribution of the target variable.
    :param learning_rate: The learning rate of the model.
    :param n_estimators_max: The maximum number of estimators to use.
    :param min_samples_split: The minimum number of samples required to split an internal node.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
    :param max_depth: The maximum depth of the tree.
    :param features: The features to use for each estimator.
    :param glm_init: Whether to initialize the model with a GLM.
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param random_state: The seed used by the random number generator.
    :param rng: The random number generator.
    :param logger: The logger to use.
    """

    if logger is None:
        logger = LocalGLMBoostLogger(verbose=0)
    if rng is None:
        rng = np.random.default_rng(random_state)
    folds = _fold_split(X=X, n_splits=n_splits, rng=rng)
    if isinstance(distribution, str):
        distribution = initiate_distribution(distribution=distribution)
    p = X.shape[1]
    n_estimators_max = (
        n_estimators_max
        if isinstance(n_estimators_max, list)
        else [n_estimators_max] * p
    )
    loss_train = np.ones((n_splits, max(n_estimators_max) + 1, p)) * np.nan
    loss_valid = np.ones((n_splits, max(n_estimators_max) + 1, p)) * np.nan

    for i, idx in enumerate(folds):
        logger.append_format_level(f"fold {i+1}/{n_splits}")
        logger.log("tuning", verbose=1)

        idx_train, idx_valid = idx
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        model = LocalGLMBooster(
            distribution=distribution,
            learning_rate=learning_rate,
            n_estimators=0,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            glm_init=glm_init,
        )
        model.fit(X_train, y_train, features=features)
        z_train = model.predict(X_train)
        z_valid = model.predict(X_valid)
        loss_train[i, 0, :] = model.distribution.loss(y=y_train, z=z_train).sum()
        loss_valid[i, 0, :] = model.distribution.loss(y=y_valid, z=z_valid).sum()

        for k in range(1, max(n_estimators_max) + 1):
            for j in range(p):
                if k < n_estimators_max[j]:
                    model.add_tree(X=X_train, y=y_train, j=j, z=z_train)
                    z_train += (
                        model.learning_rate[j]
                        * model.trees[j][-1].predict(X_train)
                        * X_train[:, j]
                    )
                    z_valid += (
                        model.learning_rate[j]
                        * model.trees[j][-1].predict(X_valid)
                        * X_valid[:, j]
                    )

                    loss_train[i, k, j] = model.distribution.loss(
                        y=y_train, z=z_train
                    ).sum()
                    loss_valid[i, k, j] = model.distribution.loss(
                        y=y_valid, z=z_valid
                    ).sum()
                else:
                    if j == 0:
                        loss_train[i, k, j] = loss_train[i, k - 1, -1]
                        loss_valid[i, k, j] = loss_valid[i, k - 1, -1]
                    else:
                        loss_train[i, k, j] = loss_train[i, k, j - 1]
                        loss_valid[i, k, j] = loss_valid[i, k, j - 1]

            if k == max(n_estimators_max):
                logger.log(
                    msg="tuning did not converge",
                    verbose=1,
                )
            elif np.all(
                [loss_valid[i, k, 0] >= loss_valid[i, k - 1, -1]]
                + [loss_valid[i, k, j] >= loss_valid[i, k, j - 1] for j in range(1, p)]
            ):
                loss_valid[i, k + 1 :, :] = loss_valid[i, k, -1]
                logger.log(
                    msg=f"tuning converged after {k} steps",
                    verbose=1,
                )
                break

            logger.log_progress(
                step=k, total_steps=max(n_estimators_max) + 1, verbose=2
            )

        logger.reset_progress()
        logger.remove_format_level()

    loss_total = loss_valid.sum(axis=0)
    loss_delta = np.zeros((p, max(n_estimators_max) + 1))
    loss_delta[0, 1:] = loss_total[1:, 0] - loss_total[:-1, -1]
    for j in range(1, p):
        loss_delta[j, 1:] = loss_total[1:, j] - loss_total[1:, j - 1]
    n_estimators = np.maximum(0, np.argmax(loss_delta > 0, axis=1) - 1)
    did_not_converge = (loss_delta > 0).sum(axis=1) == 0
    for j in range(p):
        if did_not_converge[j] and n_estimators_max[j] > 0:
            logger.log(f"tuning did not converge for coefficient {j}", verbose=1)
            n_estimators[j] = n_estimators_max[j]

    return {
        "n_estimators": n_estimators,
        "loss": {"train": loss_train, "valid": loss_valid},
    }
