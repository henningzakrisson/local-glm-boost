import os
from pathlib import Path
import yaml
import shutil
import logging
import argparse
import sys
import json
import time

import statsmodels.api as sm
import numpy as np
import pandas as pd
from rpy2.robjects import r
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators
from local_glm_boost.utils.logger import LocalGLMBoostLogger

rpy2_logger.setLevel(logging.ERROR)

# Add the parent directory to the Python path to be able
# to import the local_glm_boost package (hacky solution)
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)


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
    intercept_term = True

    return train_data, test_data, features, parallel_fit, intercept_term


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
    features = []  # Finished features
    parallel_fit = []  # Features to fit in parallel

    # Concatenate dataframes for ease of processing
    train_data["train"] = 1
    test_data["train"] = 0
    data = pd.concat([train_data, test_data], axis=0)

    # Preprocess area
    data["Area"] = data["Area"].apply(lambda x: ord(x) - 65)

    # Standardize continuous features
    train_mean = data.loc[data["train"] == 1, continuous_features].mean()
    train_std = data.loc[data["train"] == 1, continuous_features].std()
    data[continuous_features] = data[continuous_features].sub(train_mean).div(train_std)
    features += continuous_features

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

    intercept_term = False

    return train_data, test_data, features, parallel_fit, intercept_term


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
    start_time = time.time()
    intercept = LocalGLMBooster(
        n_estimators=0,
        distribution=distribution,
        glm_init=False,
    )
    intercept.fit(X=X, y=y, w=w)
    computation_time = time.time() - start_time
    return intercept, computation_time


def fit_glm(data, distribution, intercept_term):
    X, y, w = extract_data(data)
    start_time = time.time()
    glm = LocalGLMBooster(
        n_estimators=0,
        distribution=distribution,
        glm_init=True,
        intercept_term=intercept_term,
    )
    glm.fit(X=X, y=y, w=w)
    computation_time = time.time() - start_time
    return glm, computation_time


def fit_gbm(data, distribution, config, rng, logger, stratified):
    X, y, w = extract_data(data)
    start_tuning_time = time.time()
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
    tuning_time = time.time() - start_tuning_time
    start_fit_time = time.time()
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
    fit_time = time.time() - start_fit_time
    return gbm, n_estimators, tuning_loss, tuning_time, fit_time


def fit_local_glm_boost(
    data,
    distribution,
    config,
    rng,
    logger,
    stratified,
    parallel_fit,
    intercept_term,
    n_estimators_max,
):
    X, y, w = extract_data(data)
    start_tuning_time = time.time()
    local_glm_boost = LocalGLMBooster(
        n_estimators=0,
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        distribution=distribution,
        glm_init=True,
        intercept_term=intercept_term,
    )
    tuning_results = tune_n_estimators(
        X=X,
        y=y,
        w=w,
        model=local_glm_boost,
        n_estimators_max=n_estimators_max,
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
    fold_feature_importances = tuning_results["feature_importances"]
    tuning_time = time.time() - start_tuning_time

    start_fit_time = time.time()
    local_glm_boost = LocalGLMBooster(
        n_estimators=n_estimators,
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        distribution=distribution,
        glm_init=True,
    )
    local_glm_boost.fit(X=X, y=y, w=w, parallel_fit=parallel_fit)
    fit_time = time.time() - start_fit_time
    return (
        local_glm_boost,
        n_estimators,
        tuning_loss,
        tuning_time,
        fit_time,
        fold_feature_importances,
    )


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
            feature_importances.loc[feature] = (
                local_glm_boost.compute_feature_importances(feature, normalize=False)
            )
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


def create_time_table(tune_times, fit_times):
    time_table = pd.DataFrame(columns=["tune_time", "fit_time"])
    for model in ["Intercept", "GLM", "GBM", "LocalGLMboost"]:
        time_table.loc[model, "tune_time"] = tune_times[model]
        time_table.loc[model, "fit_time"] = fit_times[model]
    return time_table


def create_feature_importance_variation_table(feature_importances):
    # Calculate the variance of the feature importances over the folds
    feature_importance_std = pd.DataFrame(
        index=feature_importances[0].keys(), columns=feature_importances[0][0].keys()
    )
    # Go through all features
    for feature in feature_importance_std.index:
        # and the features that affect the coefficient functions
        for modifier_feature in feature_importance_std.columns:
            # Calculate how much the feature importance of this interaction varies over the folds
            # as a measure of feature impoortance robustness
            feature_importance_std.loc[feature, modifier_feature] = np.std(
                [
                    feature_importances[fold][feature][modifier_feature]
                    for fold in feature_importances.keys()
                ]
            )
    return feature_importance_std


def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


def create_loss_table(output_path, config):
    # Loss table
    prefix = config["data_source"]
    mse_table = pd.read_csv(f"{output_path}/loss_table.csv", index_col=0)
    mse_table.index.name = "Model"
    mse_table.columns = ["Train", "Test"]
    if prefix == "sim":
        mse_table = mse_table.drop("GBM")
        mse_table.loc["LocalGLMnet"] = [1.0023, 1.0047]
    elif prefix == "real":
        mse_table = 100 * mse_table.drop("True")
        mse_table.loc["LocalGLMnet"] = [23.728, 23.945]
    mse_table = mse_table.round(3).astype(str)
    mse_table.to_csv(f"{output_path}/plot_data/{prefix}_loss.csv")


def create_attention_plot(test_data, features, config, output_path, rng):
    prefix = config["data_source"]

    random_state = rng.integers(0, 10000)
    test_data = test_data.sample(500, random_state=random_state)

    if prefix == "sim":
        features_for_attentions = features
    elif prefix == "real":
        # Only consider the continuous features
        cat_features = ["VehGas", "VehBrand", "Region"]
        features_for_attentions = [
            feature
            for feature in features
            if not feature.startswith(tuple(cat_features))
        ]
    relevant_attentions = [f"beta_{feature}" for feature in features_for_attentions]

    with open(f"{output_path}/model_parameters.json") as f:
        model_parameters = json.load(f)

    test_data[features_for_attentions + relevant_attentions].to_csv(
        f"{output_path}/plot_data/{prefix}_attentions.csv", index=False
    )
    # GLM coefficient tex definitions for the attention plot
    glm_parameters_string = ""
    for j, feature in enumerate(features_for_attentions, start=1):
        this_glm_parameter = model_parameters["LocalGLMboost"]["beta0"][feature]
        if prefix == "sim":
            parameter_name = f"{prefix}Beta{int_to_roman(j)}"
        elif prefix == "real":
            parameter_name = f"{prefix}Beta{feature}"
        glm_parameters_string += (
            f"\\def\\{parameter_name}{{{this_glm_parameter:.3f}}}\n"
        )

    with open(f"{output_path}/plot_data/{prefix}_glm_parameters.tex", "w") as f:
        f.write(glm_parameters_string)


def save_model_parameters_table(output_path, config, train_data, features):
    prefix = config["data_source"]

    # Model parameters
    with open(f"{output_path}/model_parameters.json") as f:
        model_parameters = json.load(f)
    parameter_table = pd.DataFrame(
        index=pd.Index(["kappaValue", "betaValue"], name="parameter"), columns=features
    )
    parameter_table.loc["kappaValue"] = model_parameters["LocalGLMboost"][
        "n_estimators"
    ]
    parameter_table.loc["betaValue"] = np.array(
        list(model_parameters["LocalGLMboost"]["beta0"].values())
    ).round(3)

    if prefix == "sim":
        features_for_attentions = features
    elif prefix == "real":
        # Only consider the continuous features
        cat_features = ["VehGas", "VehBrand", "Region"]
        features_for_attentions = [
            feature
            for feature in features
            if not feature.startswith(tuple(cat_features))
        ]
    relevant_attentions = [f"beta_{feature}" for feature in features_for_attentions]

    # Calculate the variable importance and add it to this table
    parameter_table.loc[
        "variableImportance", [attention[5:] for attention in relevant_attentions]
    ] = (
        train_data[relevant_attentions].abs().mean()
        / train_data[relevant_attentions].abs().mean().sum()
    ).values
    parameter_table.loc[
        "variableImportance", [attention[5:] for attention in relevant_attentions]
    ] = parameter_table.loc[
        "variableImportance", [attention[5:] for attention in relevant_attentions]
    ].apply(lambda x: f"{x:.2f}")

    if prefix == "sim":
        # Fix the column names
        parameter_table.columns = [
            f"$x_{i}$" for i in range(1, len(parameter_table.columns) + 1)
        ]
        # Normalize the variable importance

        # Make sure all numbers have zeros in the decimal places in the beta row
        parameter_table.loc["betaValue"] = parameter_table.loc["betaValue"].apply(
            lambda x: f"{x:.2f}"
        )

    elif prefix == "real":
        # If the data is real, we need to make ranges for categorical variables
        # Make sure all numbers have zeros in the decimal places in the beta row
        parameter_table.loc["betaValue"] = parameter_table.loc["betaValue"].apply(
            lambda x: f"{x:.2f}"
        )
        cat_features = ["VehGas", "VehBrand", "Region"]
        for feature in cat_features:
            feature_dummies = [
                dummy_feature
                for dummy_feature in features
                if dummy_feature.startswith(feature)
            ]
            kappa_min = parameter_table.loc["kappaValue", feature_dummies].min()
            kappa_max = parameter_table.loc["kappaValue", feature_dummies].max()
            parameter_table.loc["kappaValue", feature] = f"({kappa_min} -- {kappa_max})"
            beta_min = parameter_table.loc["betaValue", feature_dummies].min()
            beta_max = parameter_table.loc["betaValue", feature_dummies].max()
            parameter_table.loc["betaValue", feature] = f"({beta_min} -- {beta_max})"

            # The variable importance is not calculated for categorical variables
            parameter_table.loc["variableImportance", feature] = "-"

            # Drop the dummy features
            parameter_table = parameter_table.drop(feature_dummies, axis=1)

        # Make sure the order is correct
        parameter_table = parameter_table[
            [
                "VehPower",
                "VehAge",
                "Density",
                "DrivAge",
                "BonusMalus",
                "Area",
                "VehGas",
                "VehBrand",
                "Region",
            ]
        ]

    # transpose the table
    parameter_table = parameter_table.transpose()
    # Name the index "feature"
    parameter_table.index.name = "featureName"
    parameter_table.to_csv(f"{output_path}/plot_data/{prefix}_parameters.csv")


def create_prediction_plot(config, output_path, test_data):
    prefix = config["data_source"]

    # Prediction plots
    if prefix == "sim":
        # Save mu, mu_GLM and mu_LocalGLMboost as a csv file
        test_data[["mu", "mu_GLM", "mu_LocalGLMboost"]].to_csv(
            f"{output_path}/plot_data/{prefix}_mu_hat.csv", index=False
        )

    elif prefix == "real":
        window = 1000
        # Load the entire test data
        test_data = pd.read_csv(f"{output_path}/test_data.csv", index_col=0)
        # Sort the data by the predicted z of the LocalGLMboost model
        test_data = test_data.sort_values(by="z_LocalGLMboost").reset_index(drop=True)
        # Calculate rolling averages of the predicted exp(z) for models:
        # Intercept, GLM, GBM, LocalGLMboost
        rolling_average = (
            np.exp(test_data[["z_Intercept", "z_GLM", "z_GBM", "z_LocalGLMboost"]])
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        # Name the columns after the models
        rolling_average.columns = [
            col.replace("z_", "") for col in rolling_average.columns
        ]
        # Also add rolling window of y divided by rolling window of w
        rolling_average["y"] = (
            test_data["y"].rolling(window=window, center=True, min_periods=1).mean()
            / test_data["w"].rolling(window=window, center=True, min_periods=1).mean()
        )
        # Sort so that y comes first
        rolling_average = rolling_average[
            ["y", "Intercept", "GLM", "GBM", "LocalGLMboost"]
        ]
        rolling_average.index.name = "index"
        # Sample 500 rows from the rolling average evenly
        rolling_average = rolling_average.iloc[:: len(rolling_average) // 500, :]
        # Save the rolling average as a csv file
        rolling_average.to_csv(f"{output_path}/plot_data/{prefix}_mu_hat.csv")


def save_feature_importance(config, output_path):
    prefix = config["data_source"]

    # Feature importance plot
    feature_importance = pd.read_csv(
        f"{output_path}/feature_importance.csv", index_col=0
    )
    if prefix == "real":
        # Sum the categorical features importance
        cat_features = ["VehGas", "VehBrand", "Region"]
        for feature in cat_features:
            dummy_features = [
                dummy_feature
                for dummy_feature in feature_importance.columns
                if dummy_feature.startswith(feature)
            ]
            feature_importance[feature] = feature_importance[dummy_features].sum(axis=1)
            feature_importance.loc[feature] = feature_importance.loc[
                dummy_features
            ].sum()
            # Drop the dummy features
            feature_importance = feature_importance.drop(dummy_features, axis=1)
            feature_importance = feature_importance.drop(dummy_features, axis=0)

        # Make sure order is correct
        feature_importance = feature_importance[
            [
                "VehPower",
                "VehAge",
                "Density",
                "DrivAge",
                "BonusMalus",
                "Area",
                "VehGas",
                "VehBrand",
                "Region",
            ]
        ]
    # Normalize the feature importance
    feature_importance = feature_importance.div(
        feature_importance.sum(axis=1), axis=0
    ).fillna(0)
    feature_importance.index = range(len(feature_importance))
    feature_importance.index.name = "beta"
    feature_importance.columns = range(len(feature_importance.columns))
    feature_importance = feature_importance.round(2)

    # Melt for heatmap plot
    fi_long = feature_importance.melt(
        ignore_index=False, var_name="feature"
    ).reset_index()
    fi_long = fi_long.sort_values(by=["feature", "beta"])[["feature", "beta", "value"]]
    fi_long.to_csv(
        f"{output_path}/plot_data/{prefix}_feature_importance.csv", index=False
    )


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
    if config["data_source"] == "sim":
        logger.log("Simulating data")
        train_data, test_data, features, parallel_fit, intercept_term = simulate_data(
            config["n"], output_path
        )

        distribution = "normal"

        def link(z):
            return z

        def loss_function(y, z, w):
            return (y - w * link(z)) ** 2

        stratified = False
        n_estimators_max = config["n_estimators_max"]

    elif config["data_source"] == "real":
        logger.log("Loading data")
        train_data, test_data, features, parallel_fit, intercept_term = load_data(
            config["n"], output_path, rng=rng
        )

        distribution = "poisson"

        def link(z):
            return np.exp(z)

        def loss_function(y, z, w):
            return poisson_deviance(y, w, z)

        stratified = True
        # categorical_features = [
        #     "VehBrand",
        #     "Region",
        #     "VehGas",
        # ]
        features = [
            feature
            for feature in train_data.columns
            if feature not in ["y", "z", "w", "mu"]
        ]
        # n_estimators_max should be a list that has the value config["n_estimators_max"] for each feature position
        # where the feature does not start with any of the strings in categorical_features
        n_estimators_max = [config["n_estimators_max"] for feature in features]

    else:
        raise ValueError("Data source not recognized")

    # Fit models
    logger.append_format_level("Fitting models")
    logger.log("Intercept")
    intercept, fit_time_intercept = fit_intercept(
        data=train_data, distribution=distribution
    )
    logger.log("GLM")
    glm, fit_time_glm = fit_glm(
        data=train_data, distribution=distribution, intercept_term=intercept_term
    )
    logger.log("GBM")
    gbm, n_estimators_gbm, tuning_loss_gbm, tune_time_gbm, fit_time_gbm = fit_gbm(
        data=train_data,
        distribution=distribution,
        config=config,
        rng=rng,
        logger=logger,
        stratified=stratified,
    )
    logger.log("LocalGLMboost")
    (
        local_glm_boost,
        n_estimators,
        tuning_loss,
        tune_time_local_glm_boost,
        fit_time_local_glm_boost,
        fold_feature_importances,
    ) = fit_local_glm_boost(
        data=train_data,
        distribution=distribution,
        config=config,
        rng=rng,
        logger=logger,
        stratified=stratified,
        parallel_fit=parallel_fit,
        intercept_term=intercept_term,
        n_estimators_max=n_estimators_max,
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
    time_table = create_time_table(
        tune_times={
            "Intercept": np.nan,
            "GLM": np.nan,
            "GBM": tune_time_gbm,
            "LocalGLMboost": tune_time_local_glm_boost,
        },
        fit_times={
            "Intercept": fit_time_intercept,
            "GLM": fit_time_glm,
            "GBM": fit_time_gbm,
            "LocalGLMboost": fit_time_local_glm_boost,
        },
    )
    feature_importance_variation_table = create_feature_importance_variation_table(
        fold_feature_importances
    )

    # Save data
    logger.log("Saving data")
    tuning_loss.to_csv(f"{output_path}/tuning_loss.csv")
    train_data.to_csv(f"{output_path}/train_data.csv")
    test_data.to_csv(f"{output_path}/test_data.csv")
    feature_importances.to_csv(f"{output_path}/feature_importance.csv")
    loss_table.to_csv(f"{output_path}/loss_table.csv")
    time_table.to_csv(f"{output_path}/time_table.csv")
    feature_importance_variation_table.to_csv(
        f"{output_path}/feature_importance_std.csv"
    )
    with open(f"{output_path}/model_parameters.json", "w") as json_file:
        json.dump(model_parameters, json_file)

    # Postprocess data
    logger.log("Postprocessing data")
    os.makedirs(f"{output_path}/plot_data", exist_ok=True)
    create_loss_table(output_path, config)
    create_attention_plot(test_data, features, config, output_path, rng)
    save_model_parameters_table(output_path, config, train_data, features)
    create_prediction_plot(config, output_path, test_data)

    logger.log("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the LocalGLMBoost model with specified configuration"
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    main(args.config_path)
