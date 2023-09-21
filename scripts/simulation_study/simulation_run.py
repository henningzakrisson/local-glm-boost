import os
import yaml
import shutil
import logging

import numpy as np
import pandas as pd
from rpy2.robjects import r
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
rpy2_logger.setLevel(logging.ERROR)

from local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators
from local_glm_boost.utils.logger import LocalGLMBoostLogger
from save_for_report import save_tables_and_figures

config_name = "simulation_config"

# Set up output folder, configuration file, run_id and logger
script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, f"{config_name}.yaml")
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
folder_path = os.path.join(script_dir, "../../data/output/")
save_to_git = config["save_to_git"]
if save_to_git:
    folder_path = folder_path[:-1] + "_saved/"

# Find a run_id
if os.path.exists(folder_path) and os.listdir(folder_path):
    run_id = (
        max(
            [int(folder_name.split("_")[1]) for folder_name in os.listdir(folder_path)]
            + [0]
        )
        + 1
    )
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
r_script_modified = f'output_dir <- "{output_path}"\n' + r_script
r(r_script_modified)

train_data = pd.read_csv(output_path+"train_data.csv")
test_data = pd.read_csv(output_path+"test_data.csv")

X_train = train_data.drop(columns=["Y","mu"])
y_train = train_data["Y"]
mu_train = train_data["mu"]

X_test = test_data.drop(columns=["Y","mu"])
y_test = test_data["Y"]
mu_test = test_data["mu"]

features = X_train.columns

rng = np.random.default_rng(random_state)

# Tune n_estimators
logger.log("Tuning model")
model = LocalGLMBooster(
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
    model=model,
    n_estimators_max=n_estimators_max,
    n_splits=n_splits,
    rng=rng,
    logger=logger,
    parallel=parallel,
    n_jobs=n_jobs,
)
n_estimators = tuning_results["n_estimators"]
loss = tuning_results["loss"]

logger.log("Fitting models")
model = LocalGLMBooster(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution="normal",
    glm_init=glm_init,
)
model.fit(X=X_train, y=y_train)

# Intercept model
z0 = y_train.mean()

# Summarize output
logger.log("Summarizing output")

# Save the total CV losses
loss_train = pd.DataFrame(data=np.sum(loss["train"], axis=0), columns=features)
loss_valid = pd.DataFrame(data=np.sum(loss["valid"], axis=0), columns=features)
loss_train.to_csv(f"{output_path}loss_tuning_train.csv")
loss_valid.to_csv(f"{output_path}loss_tuning_valid.csv")

# Crate a dataframe with all model predictions on test vs train data
train_data = pd.DataFrame(index=y_train.index)
train_data["y"] = y_train
train_data["mu"] = mu_train
train_data["z_0"] = np.full(len(y_train), z0)
train_data["z_glm"] = model.z0 + (model.beta0 * X_train).sum(axis=1)
train_data["z_local_glm_boost"] = model.predict(X_train)
train_data.to_csv(f"{output_path}train_data.csv")

test_data = pd.DataFrame(index=y_test.index)
test_data["y"] = y_test
test_data["mu"] = mu_test
test_data["z_0"] = np.full(len(y_test), z0)
test_data["z_glm"] = model.z0 + (model.beta0 * X_test).sum(axis=1)
test_data["z_local_glm_boost"] = model.predict(X_test)
test_data.to_csv(f"{output_path}test_data.csv")

# Create a dataframe with feature importances
feature_importances = pd.DataFrame(index=features, columns=features)
for feature in features:
    if n_estimators[feature] != 0:
        feature_importances.loc[feature] = model.compute_feature_importances(
            feature, normalize=False
        )
    else:
        feature_importances.loc[feature] = 0
feature_importances.to_csv(f"{output_path}feature_importances.csv")

# Create a dataframe with model parameters
parameters = pd.DataFrame(index=features, columns=["n_estimators", "beta0"])
for j, feature in enumerate(features):
    parameters.loc[feature] = [n_estimators[feature], model.beta0[j]]
parameters.to_csv(f"{output_path}parameters.csv")

# Save tables and figures for the report
logger.log("Saving tables and figures")
save_tables_and_figures(run_id = run_id, save_to_git = save_to_git)

logger.log("Done!")
