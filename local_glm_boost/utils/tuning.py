from typing import Union, List, Dict, Tuple, Optional
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from local_glm_boost.local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.fix_datatype import fix_datatype
from .logger import LocalGLMBoostLogger


def tune_n_estimators(
    X: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
    model: LocalGLMBooster = LocalGLMBooster(),
    n_estimators_max: Union[int, List[int]] = 1000,
    n_splits: int = 4,
    rng: Optional[np.random.Generator] = None,
    random_state: Optional[int] = None,
    logger: Optional[LocalGLMBoostLogger] = None,
    parallel: bool = True,
    n_jobs: int = -1,
    stratified: bool = False,
) -> Dict[str, Union[List[int], Dict[str, np.ndarray]]]:
    """Tunes the n_estimators parameter of a LocalGBMBoost model using k-fold cross-validation.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target values of shape (n_samples,).
    :param w: The weights of the observations. If `None`, all weights are set to 1.
    :param model: The GBM model to tune.
    :param n_estimators_max: The maximum number of estimators to try.
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param rng: The random number generator.
    :param random_state: The random state. Only used if rng is None.
    :param logger: The simulation logger to use for logging.
    :param parallel: Whether to use parallel processing for the cross-validation.
    :param n_jobs: The number of jobs to use for parallel processing. Default is -1, which uses all available cores.
    :param stratified: Whether to use stratified k-fold cross-validation.
    """
    logger = LocalGLMBoostLogger(verbose=-1) if logger is None else logger
    rng = np.random.default_rng(random_state) if rng is None else rng
    model.reset(n_estimators=0)

    n_estimators_max = (
        n_estimators_max
        if isinstance(n_estimators_max, list)
        else [n_estimators_max] * X.shape[1]
    )

    w = np.ones_like(y) if w is None else w
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    X, y, w = fix_datatype(X=X, y=y, w=w)
    folds = _fold_split(
        X=X, y=y, w=w, n_splits=n_splits, rng=rng, stratified=stratified
    )

    logger.log(f"performing cross-validation on {n_splits} folds")
    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_fold)(
                fold=folds[i],
                model=model,
                n_estimators_max=n_estimators_max,
            )
            for i in folds
        )
    else:
        results = []
        for i in folds:
            logger.log(f"fold {i+1}/{n_splits}")
            results.append(
                _evaluate_fold(
                    fold=folds[i],
                    model=model,
                    n_estimators_max=n_estimators_max,
                )
            )

    loss = {
        "train": [result[0] for result in results],
        "valid": [result[1] for result in results],
    }

    n_estimators = _find_n_estimators(
        loss=np.sum(loss["valid"], axis=0),
        n_estimators_max=n_estimators_max,
        logger=logger,
    )

    if "feature_names" in locals():
        n_estimators = pd.Series(n_estimators, index=feature_names)

    return {
        "n_estimators": n_estimators,
        "loss": loss,
    }


def _fold_split(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
    stratified: bool = False,
) -> Dict[
    int,
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Split data into k folds.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target values of shape (n_samples,).
    :param w: The weights of the observations.
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param rng: The random number generator.
    :return A dictionary containing the folds as tuples in the order
        (X_train, y_train, X_valid, y_valid).
    """
    random_state = rng.integers(low=0, high=2**32 - 1)  # Generate a random integer

    if stratified:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    idx_folds = [test_index for _, test_index in splitter.split(X, y)]
    folds = {}
    for i in range(n_splits):
        idx_test = idx_folds[i]
        idx_train = np.concatenate(idx_folds[:i] + idx_folds[i + 1 :])
        folds[i] = (
            X[idx_train],
            y[idx_train],
            w[idx_train],
            X[idx_test],
            y[idx_test],
            w[idx_test],
        )
    return folds


def _evaluate_fold(
    fold: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    model: LocalGLMBooster,
    n_estimators_max: List[int],
):
    X_train, y_train, w_train, X_valid, y_valid, w_valid = fold

    model.fit(X_train, y_train, w_train)
    z_train = model.predict(X_train)
    z_valid = model.predict(X_valid)

    loss_train = np.zeros((max(n_estimators_max) + 1, model.p))
    loss_valid = np.zeros((max(n_estimators_max) + 1, model.p))
    loss_train[0, :] = model.distribution.loss(y=y_train, z=z_train, w=w_train).sum()
    loss_valid[0, :] = model.distribution.loss(y=y_valid, z=z_valid, w=w_valid).sum()

    for k in range(1, max(n_estimators_max) + 1):
        for j in range(model.p):
            if k < n_estimators_max[j]:
                tree = model.fit_tree(X=X_train, y=y_train, z=z_train, w=w_train, j=j)
                model.trees[j].append(tree)
                model.n_estimators[j] += 1

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

                loss_train[k, j] = model.distribution.loss(
                    y=y_train, z=z_train, w=w_train
                ).sum()
                loss_valid[k, j] = model.distribution.loss(
                    y=y_valid, z=z_valid, w=w_valid
                ).sum()
            else:
                loss_train[k, j] = (
                    loss_train[k - 1, -1] if j == 0 else loss_train[k, j - 1]
                )
                loss_valid[k, j] = (
                    loss_valid[k - 1, -1] if j == 0 else loss_valid[k, j - 1]
                )

        if _has_tuning_converged(
            current_loss=loss_valid[k],
            previous_loss=loss_valid[k - 1],
        ):
            loss_train[k + 1 :, :] = loss_train[k, -1]
            loss_valid[k + 1 :, :] = loss_valid[k, -1]
            break

    return loss_train, loss_valid


def _has_tuning_converged(
    current_loss: np.ndarray,
    previous_loss: np.ndarray,
) -> bool:
    """Check if the tuning has converged after a complete boosting iteration.

    :param current_loss: The current loss for all parameter dimensions.
    :param previous_loss: The previous loss for all parameter dimensions.
    :return: True if the tuning has converged, False otherwise.
    """
    loss_delta = np.zeros_like(current_loss)
    loss_delta[1:] = current_loss[1:] - current_loss[:-1]
    loss_delta[0] = current_loss[0] - previous_loss[-1]
    return np.all(loss_delta >= 0)


def _find_n_estimators(
    loss: np.ndarray,
    n_estimators_max: Union[int, List[int]],
    logger: LocalGLMBoostLogger,
) -> List[int]:
    """Find the number of estimators for each parameter dimension.

    :param loss: The loss for each parameter dimension and number of estimators.
    :param n_estimators_max: The maximum number of estimators for each parameter dimension.
    :param logger: The logger.
    :return: The number of estimators for each parameter dimension.
    """
    loss_delta = np.zeros_like(loss)
    loss_delta[1:, 0] = loss[1:, 0] - loss[:-1, -1]
    loss_delta[1:, 1:] = loss[1:, 1:] - loss[1:, :-1]

    n_estimators = np.maximum(0, np.argmax(loss_delta[1:] > 0, axis=0))
    did_not_converge = (loss_delta > 0).sum(axis=0) == 0
    n_estimators[did_not_converge] = np.array(n_estimators_max)[did_not_converge]
    if np.any(did_not_converge):
        logger.log(
            f"tuning did not converge for dimensions {np.where(did_not_converge)}",
            verbose=1,
        )
    return list(n_estimators)
