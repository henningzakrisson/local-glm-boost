import os
import yaml
import shutil
import logging

import numpy as np
import pandas as pd
from rpy2.robjects import r
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators
from local_glm_boost.utils.logger import LocalGLMBoostLogger
from local_glm_boost.utils.fix_data import fix_data

rpy2_logger.setLevel(logging.ERROR)
config_name = "simulation_config"

# Set up output folder, configuration file, run_id and logger
script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, f"{config_name}.yaml")
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
folder_path = os.path.join(script_dir, "../../data/output/")

# Find a run_id
if os.path.exists(folder_path) and os.listdir(folder_path):
    run_ids = []
    for folder_name in os.listdir(folder_path):
        try:
            prefix, run_number = folder_name.split("_", 1)  # Split by the first underscore
            if prefix == "run" and run_number.isdigit():  # Check if prefix is "run" and run_number is a digit
                run_ids.append(int(run_number))
        except ValueError:
            continue  # Skip to the next folder_name if it can't be split into two parts
    run_id = max(run_ids + [0]) + 1
else:
    run_id = 0

output_path = os.path.join(folder_path, f"run_{run_id}/")
os.makedirs(output_path)

# Put the config in the output folder
shutil.copyfile(config_path, f"{output_path}config.yaml")

# Set up logger
logger = LocalGLMBoostLogger(
    verbose=2,
    output_path=output_path,
)
logger.append_format_level(f"run_{run_id}")

logger.log("Loading configuration")
n = config["n"]
distribution = config["distribution"]
learning_rate = config["learning_rate"]
n_estimators_max = config["n_estimators_max"]
min_samples_split = config["min_samples_split"]
min_samples_leaf = config["min_samples_leaf"]
max_depth = config["max_depth"]
glm_init = config["glm_init"]
feature_selection = config["feature_selection"]
n_splits = config["n_splits"]
parallel = config["parallel"]
n_jobs = config["n_jobs"]
random_state = config["random_state"]

logger.log("Simulating data")
with open("simulate_data.R", 'r') as file:
    r_script = file.read()
# Add number of data points and output path to R script
r_script = f'N <- {n}\n' + r_script
r_script = f'output_dir <- "{output_path}"\n' + r_script
r(r_script)

train_data = pd.read_csv(output_path+"train_data.csv")
test_data = pd.read_csv(output_path+"test_data.csv")
train_data["w"] = 1
test_data["w"] = 1

# Generic part of script below (i.e. should be data agnostic)
def extract_data(df):
    X = df.drop(columns=["Y", "mu"], errors='ignore')
    y = df["Y"]
    mu = df["mu"] if "mu" in df.columns else pd.Series(np.ones(len(y)) * np.nan)
    w = df["w"] if "w" in df.columns else pd.Series(np.ones(len(y)))
    return X, y, mu, w

X_train, y_train, mu_train, w_train = extract_data(train_data)
X_test, y_test, mu_test, w_test = extract_data(test_data)

# Save feature names
features = X_train.columns
# Initiate random number generator
rng = np.random.default_rng(random_state)

# Fit an intercept model
logger.log("Fitting intercept model")
z0 = y_train.mean()

# Fit a GLM model
logger.log("Fitting GLM model")
glm = LocalGLMBooster(
    n_estimators=0,
    distribution=distribution,
    glm_init=glm_init,
)
glm.fit(X=X_train, y=y_train, w=w_train)

# Add a standard GBM
logger.log("Tuning GBM")
def add_constant_column(df):
    df['const'] = 1
    return df[['const'] + [col for col in df.columns if col != 'const']]

X_train_const = add_constant_column(X_train.copy())
X_test_const = add_constant_column(X_test.copy())

model_standard = LocalGLMBooster(
    distribution=distribution,
    n_estimators=0,
    learning_rate=learning_rate,
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    max_depth=max_depth,
    glm_init=False,
)

tuning_results_gbm = tune_n_estimators(
    X=X_train_const,
    y=y_train,
    w=w_train,
    model=model_standard,
    n_estimators_max=[n_estimators_max]+[0] * len(features),
    rng=rng,
    n_splits=n_splits,
    parallel=parallel,
    n_jobs=n_jobs,
    logger=logger,
)
n_estimators_gbm = tuning_results_gbm["n_estimators"]
loss_gbm = tuning_results_gbm["loss"]

logger.log("Fitting GBM model")
gbm = LocalGLMBooster(
    n_estimators=n_estimators_gbm,
    learning_rate=learning_rate,
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    max_depth=max_depth,
    distribution=distribution,
    glm_init=False,
)
gbm.fit(X=X_train_const, y=y_train, w=w_train)

# Add the LocalGLMboost model
logger.log("Tuning LocalGLMboost model")
local_glm_boost = LocalGLMBooster(
    n_estimators=0,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution=distribution,
    glm_init=glm_init,
    feature_selection=feature_selection,
)
tuning_results = tune_n_estimators(
    X=X_train,
    y=y_train,
    w=w_train,
    model=local_glm_boost,
    n_estimators_max=n_estimators_max,
    n_splits=n_splits,
    rng=rng,
    logger=logger,
    parallel=parallel,
    n_jobs=n_jobs,
)
n_estimators = tuning_results["n_estimators"]
loss = tuning_results["loss"]

logger.log("Fitting LocalGLMboost model")
local_glm_boost = LocalGLMBooster(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution="normal",
    glm_init=glm_init,
)
local_glm_boost.fit(X=X_train, y=y_train,w=w_train)

# Summarize output
logger.log("Summarizing output")

# Save the tuning losses
tuning_loss_train = pd.DataFrame(data=np.sum(loss["train"], axis=0), columns=features)
tuning_loss_valid = pd.DataFrame(data=np.sum(loss["valid"], axis=0), columns=features)
tuning_loss_train["gbm"] = pd.DataFrame(data=np.sum(loss_gbm["train"], axis=0)[:,0], columns=["gbm"])
tuning_loss_valid["gbm"] = pd.DataFrame(data=np.sum(loss_gbm["valid"], axis=0)[:,0], columns=["gbm"])
# Combine these into a multiindex dataframe
tuning_loss = pd.concat([tuning_loss_train, tuning_loss_valid], axis=1, keys=["train", "valid"])
tuning_loss.to_csv(f"{output_path}tuning_loss.csv")

# Add predictions to the training data
train_data["z_intercept"] = np.full(len(y_train), z0)
train_data["z_glm"] = glm.predict(X_train)
train_data["z_local_glm_boost"] = local_glm_boost.predict(X_train)
# Add regression attentions
X_train_fixed = fix_data(X=X_train, feature_names=local_glm_boost.feature_names)
beta_hat = local_glm_boost.predict_parameter(X=X_train_fixed)
for j, feature in enumerate(features):
    train_data[f"beta_{feature}"] = beta_hat[j]
# Save data
train_data.to_csv(f"{output_path}train_data.csv")

# Add predictions to the test data
test_data["z_intercept"] = np.full(len(y_test), z0)
test_data["z_glm"] = glm.predict(X_test)
test_data["z_local_glm_boost"] = local_glm_boost.predict(X_test)
# Add regression attentions
X_test_fixed = fix_data(X=X_test, feature_names=local_glm_boost.feature_names)
beta_hat = local_glm_boost.predict_parameter(X=X_test_fixed)
for j, feature in enumerate(features):
    test_data[f"beta_{feature}"] = beta_hat[j]
# Save data
test_data.to_csv(f"{output_path}test_data.csv")

# Create a dataframe with feature importance scores
feature_importances = pd.DataFrame(index=features, columns=features)
for feature in features:
    if n_estimators[feature] != 0:
        feature_importances.loc[feature] = local_glm_boost.compute_feature_importances(
            feature, normalize=False
        )
    else:
        feature_importances.loc[feature] = 0
feature_importances.to_csv(f"{output_path}feature_importance.csv")

# Create a dataframe with model parameters
parameters = pd.DataFrame(index=features, columns=["n_estimators", "beta0","beta_glm"])
for j, feature in enumerate(features):
    parameters.loc[feature] = [n_estimators[feature], local_glm_boost.beta0[j], glm.beta0[j]]
parameters.loc["intercept"] = [np.nan, local_glm_boost.z0, glm.z0]
parameters.to_csv(f"{output_path}parameters.csv")

# Finally create a simple loss table
loss_data = {"train": {}, "test": {}}
for y, w, X, X_const, mu, data_label in zip(
    [y_train, y_test], [w_train, w_test],
    [X_train, X_test], [X_train_const, X_test_const],
    [mu_train, mu_test], ["train", "test"]
):
    loss_data[data_label]["true"] = local_glm_boost.distribution.loss(y=y, z=mu, w=w).mean()
    loss_data[data_label]["intercept"] = local_glm_boost.distribution.loss(y=y, z=z0, w=w).mean()
    loss_data[data_label]["glm"] = local_glm_boost.distribution.loss(y=y, z=glm.predict(X), w=w).mean()
    loss_data[data_label]["gbm"] = local_glm_boost.distribution.loss(y=y, z=gbm.predict(X_const), w=w).mean()
    loss_data[data_label]["local_glm_boost"] = local_glm_boost.distribution.loss(y=y, z=local_glm_boost.predict(X), w=w).mean()

# Create DataFrame from the dictionary
loss_table = pd.DataFrame.from_dict(loss_data, orient='index')

# Save to CSV
loss_table.to_csv(f"{output_path}loss_table.csv")


logger.log("Done!")
