import numpy as np
import pandas as pd
import os
import yaml
import shutil
import logging

from rpy2.robjects import r
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
rpy2_logger.setLevel(logging.ERROR)

from local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators
from local_glm_boost.utils.logger import LocalGLMBoostLogger

config_name = "real_data_config"

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

# Load stuff from the config
logger.log("Loading configuration")
n_train = config["n_train"]
features_to_use = config["features_to_use"]
target = config["target"]
weights = config["weights"]
distribution = config["distribution"]
n_estimators_max = config["n_estimators_max"]
learning_rate = config["learning_rate"]
min_samples_split = config["min_samples_split"]
min_samples_leaf = config["min_samples_leaf"]
max_depth = config["max_depth"]
glm_init = config["glm_init"]
random_seed = config["random_seed"]
n_splits = config["n_splits"]
test_size = config["test_size"]
parallel = config["parallel"]
stratified = config["stratified"]
n_jobs = config["n_jobs"]

# Load and preprocess data
logger.log("Loading data")
with open('load_data.R', 'r') as file:
    r_script = file.read()
r_script_modified = f'output_dir <- "{output_path}"\n' + r_script
r(r_script_modified)

df_train = pd.read_csv(output_path + 'train_data.csv', index_col=0)
df_test = pd.read_csv(output_path + 'test_data.csv', index_col=0)

df_train["train"] = 1
df_test["train"] = 0
df = pd.concat([df_train, df_test], axis=0)

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
parallel_fit = []

features = [feature for feature in continuous_features if feature in features_to_use]

df["Area"] = df["Area"].apply(lambda x: ord(x) - 65)

for feature in categorical_features:
    if feature in features_to_use:
        dummies = pd.get_dummies(df[feature], prefix=feature)
        df = pd.concat([df, dummies], axis=1)
        dummy_feature_indices = [
            j for j in range(len(features), len(features) + len(dummies.columns))
        ]
        parallel_fit.append(dummy_feature_indices)
        features += dummies.columns.tolist()

for feature in continuous_features:
    if feature in features_to_use:
        df[feature] = df[feature] / df.loc[df["train"] == 1, feature].max()

# Re-split train and test
df_train = df.loc[df["train"] == 1]
df_test = df.loc[df["train"] == 0]

rng = np.random.default_rng(seed=random_seed)
if n_train != "all":
    df_train = df_train.sample(n_train, random_state=rng.integers(0, 10000))
n = len(df_train)+len(df_test)

X_train = df_train[features].astype(float)
y_train = df_train[target]
w_train = df_train[weights]

X_test = df_test[features].astype(float)
y_test = df_test[target]
w_test = df_test[weights]

logger.log(f"Training set size: {len(X_train)}, test set size: {len(X_test)}")
logger.log(f"Number of features: {len(features)} ({len(continuous_features)} continuous, {len(categorical_features)} categorical)")

# Tune n_estimators
logger.log("Tuning model")

# Process the n_estimators_max hyperparameter
if isinstance(n_estimators_max, dict):
    # Find the position of every key in the features list and
    # create a list of ints with the values
    n_estimators_dict = n_estimators_max.copy()
    n_estimators_max = [0] * len(features)
    for key, value in n_estimators_dict.items():
        n_estimators_max[features.index(key)] = value

model = LocalGLMBooster(
    distribution=distribution,
    n_estimators=0,
    learning_rate=learning_rate,
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    max_depth=max_depth,
    glm_init=glm_init,
)

tuning_results = tune_n_estimators(
    X=X_train,
    y=y_train,
    w=w_train,
    model=model,
    n_estimators_max=n_estimators_max,
    rng=rng,
    n_splits=n_splits,
    parallel=parallel,
    n_jobs=n_jobs,
    stratified=stratified,
    logger=logger,
    parallel_fit=parallel_fit,
)
n_estimators = tuning_results["n_estimators"]
loss = tuning_results["loss"]

# Fit models
logger.log("Fitting models")
# Standard model
model = LocalGLMBooster(
    distribution="poisson",
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    min_samples_leaf=min_samples_leaf,
    max_depth=max_depth,
)
model.fit(X_train, y_train, w_train, parallel_fit=parallel_fit)

# Intercept model
z0 = np.log((y_train.sum() / w_train.sum()))

# Summarize output
logger.log("Summarizing output")

# Save the total CV losses
loss_train = pd.DataFrame(data=np.sum(loss["train"], axis=0), columns=features)
loss_valid = pd.DataFrame(data=np.sum(loss["valid"], axis=0), columns=features)
loss_train.to_csv(f"{output_path}loss_tuning_train.csv")
loss_valid.to_csv(f"{output_path}loss_tuning_valid.csv")

# Crate a dataframe with all model predictions on test vs train data
test_data = pd.DataFrame(index=y_test.index)
test_data["y"] = y_test
test_data["w"] = w_test
test_data["z_0"] = np.full(len(y_test), z0)
test_data["z_glm"] = model.z0 + (model.beta0 * X_test).sum(axis=1)
test_data["z_local_glm_boost"] = model.predict(X_test)
test_data.to_csv(f"{output_path}test_data.csv")

train_data = pd.DataFrame(index=y_train.index)
train_data["y"] = y_train
train_data["w"] = w_train
train_data["z_0"] = np.full(len(y_train), z0)
train_data["z_glm"] = model.z0 + (model.beta0 * X_train).sum(axis=1)
train_data["z_local_glm_boost"] = model.predict(X_train)
train_data.to_csv(f"{output_path}train_data.csv")

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
from save_for_report import save_tables_and_figures
save_tables_and_figures(run_id = run_id, save_to_git = save_to_git)

logger.log("Done!")
