import os
from pathlib import Path
import yaml
import shutil
import logging
import argparse
import sys
import json

import statsmodels.api as sm
import numpy as np
import pandas as pd
from rpy2.robjects import r
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

rpy2_logger.setLevel(logging.ERROR)

# Add the parent directory to the Python path to be able
# to import the local_glm_boost package (hacky solution)
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators
from local_glm_boost.utils.logger import LocalGLMBoostLogger
from local_glm_boost.utils.distributions import initiate_distribution


def setup_output_folder():
    """
    Sets up the output folder and determines the run_id.
    :param base_path: The base directory where the output folder will be created.
    :return: Path to the newly created output folder and the run_id.
    """
    # Create the output folder path
    base_path = Path(__file__).resolve().parent.parent
    folder_path = base_path / "data" / "output"
    os.makedirs(folder_path, exist_ok=True)

    # Determine the run_id
    existing_run_ids = [
        d for d in os.listdir(folder_path) if d.startswith("run") and d[4:].isdigit()
    ]
    run_id = max([int(run.split("_")[1]) for run in existing_run_ids], default=0) + 1

    # Create a specific folder for this run
    output_path = folder_path / f"run_{run_id}"
    os.makedirs(output_path)

    return output_path, run_id


def simulate_data(n, output_path):
    with open(f"{script_dir}/r_scripts/simulate_data.R", "r") as file:
        r_script = file.read()
    # Add number of data points and output path to R script
    r_script = f"N <- {int(np.floor(n/2))}\n" + r_script
    r_script = f'output_dir <- "{output_path}"\n' + r_script
    r(r_script)

    train_data = pd.read_csv(f"{output_path}/train_data.csv")
    test_data = pd.read_csv(f"{output_path}/test_data.csv")
    train_data["w"] = 1
    test_data["w"] = 1

    features = [col for col in train_data.columns if col not in ["y", "w", "z", "mu"]]
    parallel_fit = []

    return train_data, test_data, features, parallel_fit


def process_data(train_data, test_data, n, rng):
    if n != "all":
        train_data = train_data.sample(n, random_state=rng.integers(0, 10000))

    train_data["train"] = 1
    test_data["train"] = 0
    data = pd.concat([train_data, test_data], axis=0)
    data["z"] = np.nan

    data.set_index("IDpol", inplace=True)
    data.rename(
        columns={
            "Exposure": "w",
            "ClaimNb": "y",
        },
        inplace=True,
    )
    data.drop(columns=["ClaimTotal", "Unnamed: 0"], inplace=True)

    continuous_features = [
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density",
        "Area",
    ]
    categorical_features = [
        "VehBrand",
        "Region",
        "VehGas",
    ]
    features = continuous_features
    parallel_fit = []  # Features to fit in parallel

    # Concatenate dataframes for ease of processing
    train_data["train"] = 1
    test_data["train"] = 0
    data = pd.concat([train_data, test_data], axis=0)

    # Preprocess area
    data["Area"] = data["Area"].apply(lambda x: ord(x) - 65)

    # One-hot encode categorical features
    for feature in categorical_features:
        dummies = pd.get_dummies(data[feature], prefix=feature)
        data = pd.concat([data, dummies], axis=1)
        data.drop(columns=[feature], inplace=True)
        dummy_feature_indices = [
            j for j in range(len(features), len(features) + len(dummies.columns))
        ]
        parallel_fit.append(dummy_feature_indices)
        features += dummies.columns.tolist()

    # Standardize all features
    train_mean = data.loc[data["train"] == 1, features].mean()
    train_std = data.loc[data["train"] == 1, features].std()
    data[features] = data[features].sub(train_mean).div(train_std)

    data.rename(
        columns={
            "ClaimNb": "y",
            "Exposure": "w",
        },
        inplace=True,
    )
    data["z"] = np.nan

    # Re-split
    train_data = data.loc[data["train"] == 1, features + ["y", "w", "z"]]
    test_data = data.loc[data["train"] == 0, features + ["y", "w", "z"]]

    # TODO: Replace with actual
    parallel_fit = []
    return train_data, test_data, features, parallel_fit


def load_data(n, output_path, rng):
    with open(f"{script_dir}/r_scripts/load_data.R", "r") as file:
        r_script = file.read()
    # Add output path to R script
    r_script = f'output_dir <- "{output_path}"\n' + r_script
    r(r_script)

    train_data, test_data, features, parallel_fit = process_data(
        train_data=pd.read_csv(f"{output_path}/train_data.csv"),
        test_data=pd.read_csv(f"{output_path}/test_data.csv"),
        n=n,
        rng=rng,
    )

    return train_data, test_data, features, parallel_fit


# Calculate deviance
def poisson_deviance(y, w, z):
    log_y = np.zeros(len(y))
    log_y[y > 0] = np.log(y[y > 0])
    dev = w * np.exp(z) + y * (log_y - np.log(w) - z - 1)
    return 2 * dev


def extract_data(df):
    X = df.drop(columns=["y", "z", "w"], errors="ignore").astype(float)
    y = df["y"].astype(float)
    w = df["w"].astype(float)
    return X, y, w


def fit_intercept(data, distribution):
    X, y, w = extract_data(data)
    intercept = LocalGLMBooster(
        n_estimators=0,
        distribution=distribution,
        glm_init=False,
    )
    intercept.fit(X=X, y=y, w=w)
    return intercept


def fit_glm(data, distribution):
    X, y, w = extract_data(data)
    glm = LocalGLMBooster(
        n_estimators=0,
        distribution=distribution,
        glm_init=True,
    )
    glm.fit(X=X, y=y, w=w)
    return glm


def fit_gbm(data, distribution, config, rng, logger, stratified):
    X, y, w = extract_data(data)
    model_standard = LocalGLMBooster(
        distribution=distribution,
        n_estimators=0,
        learning_rate=config["learning_rate"],
        min_samples_leaf=config["min_samples_leaf"],
        min_samples_split=config["min_samples_split"],
        max_depth=config["max_depth"],
        glm_init=False,
    )
    tuning_results_gbm = tune_n_estimators(
        X=sm.add_constant(X.copy()),
        y=y,
        w=w,
        model=model_standard,
        n_estimators_max=[config["n_estimators_max"]] + [0] * len(X.columns),
        rng=rng,
        n_splits=config["n_splits"],
        parallel=config["parallel"],
        n_jobs=config["n_jobs"],
        logger=logger,
        stratified=stratified,
    )
    n_estimators = tuning_results_gbm["n_estimators"]
    tuning_loss = tuning_results_gbm["loss"]

    gbm = LocalGLMBooster(
        n_estimators=n_estimators,
        learning_rate=config["learning_rate"],
        min_samples_leaf=config["min_samples_leaf"],
        min_samples_split=config["min_samples_split"],
        max_depth=config["max_depth"],
        distribution=distribution,
        glm_init=False,
    )
    gbm.fit(X=sm.add_constant(X.copy()), y=y, w=w)
    return gbm, n_estimators, tuning_loss


def fit_local_glm_boost(
    data, distribution, config, rng, logger, stratified, parallel_fit
):
    X, y, w = extract_data(data)
    local_glm_boost = LocalGLMBooster(
        n_estimators=0,
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        distribution=distribution,
        glm_init=True,
    )
    tuning_results = tune_n_estimators(
        X=X,
        y=y,
        w=w,
        model=local_glm_boost,
        n_estimators_max=config["n_estimators_max"],
        n_splits=config["n_splits"],
        rng=rng,
        logger=logger,
        parallel=config["parallel"],
        n_jobs=config["n_jobs"],
        stratified=stratified,
        parallel_fit=parallel_fit,
    )
    n_estimators = tuning_results["n_estimators"]
    tuning_loss = tuning_results["loss"]

    local_glm_boost = LocalGLMBooster(
        n_estimators=n_estimators,
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        distribution=distribution,
        glm_init=True,
    )
    local_glm_boost.fit(X=X, y=y, w=w, parallel_fit=parallel_fit)
    return local_glm_boost, n_estimators, tuning_loss


def consolidate_tuning_loss(tuning_loss, tuning_loss_gbm, features):
    tuning_loss_train = pd.DataFrame(
        data=np.sum(tuning_loss["train"], axis=0), columns=features
    )
    tuning_loss_train["gbm"] = pd.DataFrame(
        data=np.sum(tuning_loss_gbm["train"], axis=0)[:, 0], columns=["gbm"]
    )
    tuning_loss_valid = pd.DataFrame(
        data=np.sum(tuning_loss["valid"], axis=0), columns=features
    )
    tuning_loss_valid["gbm"] = pd.DataFrame(
        data=np.sum(tuning_loss_gbm["valid"], axis=0)[:, 0], columns=["gbm"]
    )
    tuning_loss = pd.concat(
        [tuning_loss_train, tuning_loss_valid], axis=1, keys=["train", "valid"]
    )
    return tuning_loss


def add_predictions(data, models, link, features):
    X, y, w = extract_data(data)
    # Mean predictions
    for model in models:
        if model == "GBM":
            data[f"z_{model}"] = models[model].predict(sm.add_constant(X.copy()))
            data[f"mu_{model}"] = w * link(data[f"z_{model}"])
        else:
            data[f"z_{model}"] = models[model].predict(X=X)
            data[f"mu_{model}"] = w * link(data[f"z_{model}"])

    # True parameter
    data["mu"] = w * link(data["z"])

    # Regression attentions
    beta_hat = models["LocalGLMboost"].predict_parameter(X=X)
    for j, feature in enumerate(features):
        data[f"beta_{feature}"] = beta_hat[j]

    return data


def calculate_feature_importance(local_glm_boost, features):
    feature_importances = pd.DataFrame(index=features, columns=features)
    for j, feature in enumerate(features):
        if local_glm_boost.n_estimators[j] != 0:
            feature_importances.loc[
                feature
            ] = local_glm_boost.compute_feature_importances(feature, normalize=False)
        else:
            feature_importances.loc[feature] = 0

    return feature_importances


def save_model_parameters(models, features):
    model_parameters = {}
    model_parameters["Intercept"] = {"intercept": float(models["Intercept"].z0)}
    model_parameters["GLM"] = {
        "intercept": float(models["GLM"].z0),
        "beta0": {
            feature: float(beta0)
            for feature, beta0 in zip(features, models["GLM"].beta0)
        },
    }

    model_parameters["GBM"] = {
        "intercept": float(models["GBM"].z0),
        "n_estimators": int(models["GBM"].n_estimators[0]),
    }

    model_parameters["LocalGLMboost"] = {
        "intercept": float(models["LocalGLMboost"].z0),
        "beta0": {
            feature: float(beta0)
            for feature, beta0 in zip(features, models["LocalGLMboost"].beta0)
        },
        "n_estimators": {
            feature: int(n_estimators)
            for feature, n_estimators in zip(
                features, models["LocalGLMboost"].n_estimators.values()
            )
        },
    }
    return model_parameters


def calculate_loss_results(train_data, test_data, loss_function):
    loss_table = pd.DataFrame(columns=["train", "test"])
    for data_label, data in zip(["train", "test"], [train_data, test_data]):
        loss_table.loc["True", data_label] = loss_function(
            y=data["y"], z=data["z"], w=data["w"]
        ).mean()
        for model in ["Intercept", "GLM", "GBM", "LocalGLMboost"]:
            loss_table.loc[model, data_label] = loss_function(
                y=data["y"], z=data[f"z_{model}"], w=data["w"]
            ).mean()
    return loss_table


def main(config_path):
    # Set up output folder, run_id, and logger
    output_path, run_id = setup_output_folder()
    logger = LocalGLMBoostLogger(
        verbose=2,
        output_path=output_path,
    )
    logger.append_format_level(f"run_{run_id}")

    # Load configuration
    logger.log("Loading configuration")
    shutil.copyfile(config_path, f"{output_path}/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initiate random number generator
    # NOTE: This is not used in the simulation
    # To adjust the random state of the simulation, adjust the R script
    rng = np.random.default_rng(config["random_state"])

    # Load data
    if config["data_source"] == "simulation":
        logger.log("Simulating data")
        train_data, test_data, features, parallel_fit = simulate_data(
            config["n"], output_path
        )

        distribution = "normal"
        link = lambda z: z
        loss_function = lambda y, z, w: (y - w * link(z)) ** 2
        stratified = False

    elif config["data_source"] == "real":
        logger.log("Loading data")
        train_data, test_data, features, parallel_fit = load_data(
            config["n"], output_path, rng=rng
        )

        distribution = "poisson"
        link = lambda z: np.exp(z)
        loss_function = lambda y, z, w: poisson_deviance(y, w, z)
        stratified = True
    else:
        raise ValueError("Data source not recognized")

    # Fit models
    logger.append_format_level("Fitting models")
    logger.log("Intercept")
    intercept = fit_intercept(data=train_data, distribution=distribution)
    logger.log("GLM")
    glm = fit_glm(data=train_data, distribution=distribution)
    logger.log("GBM")
    gbm, n_estimators_gbm, tuning_loss_gbm = fit_gbm(
        data=train_data,
        distribution=distribution,
        config=config,
        rng=rng,
        logger=logger,
        stratified=stratified,
    )
    logger.log("LocalGLMboost")
    local_glm_boost, n_estimators, tuning_loss = fit_local_glm_boost(
        data=train_data,
        distribution=distribution,
        config=config,
        rng=rng,
        logger=logger,
        stratified=stratified,
        parallel_fit=parallel_fit,
    )
    models = {
        "Intercept": intercept,
        "GLM": glm,
        "GBM": gbm,
        "LocalGLMboost": local_glm_boost,
    }
    logger.remove_format_level()

    # Summarize output
    logger.log("Summarizing output")
    features = [
        feature
        for feature in train_data.columns
        if feature not in ["y", "z", "w", "mu"]
    ]
    tuning_loss = consolidate_tuning_loss(tuning_loss, tuning_loss_gbm, features)
    train_data = add_predictions(train_data, models, link, features)
    test_data = add_predictions(test_data, models, link, features)
    feature_importances = calculate_feature_importance(local_glm_boost, features)
    model_parameters = save_model_parameters(models, features)
    loss_table = calculate_loss_results(train_data, test_data, loss_function)

    # Save data
    logger.log("Saving data")
    tuning_loss.to_csv(f"{output_path}/tuning_loss.csv")
    train_data.to_csv(f"{output_path}/train_data.csv")
    test_data.to_csv(f"{output_path}/test_data.csv")
    feature_importances.to_csv(f"{output_path}/feature_importance.csv")
    loss_table.to_csv(f"{output_path}/loss_table.csv")
    with open(f"{output_path}/model_parameters.json", "w") as json_file:
        json.dump(model_parameters, json_file)

    logger.log("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the LocalGLMBoost model with specified configuration"
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    main(args.config_path)
