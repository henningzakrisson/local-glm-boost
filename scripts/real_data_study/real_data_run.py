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
to_minimize = lambda z: model.distribution.loss(y=y_train, z=z, w=w_train).mean()
z0 = np.log((y_train.sum() / w_train.sum()))
from scipy.optimize import minimize

res = minimize(
    to_minimize, z0, method="nelder-mead", options={"xatol": 1e-8, "disp": False}
)
z_opt = res.x

# Summarize output
logger.log("Summarizing output")


# Define the Poisson deviance
def deviance(y,w,z):
    log_y = np.zeros(len(y))
    log_y[y>0] = np.log(y[y>0])

    dev = 2 * (w*np.exp(z) + y*(log_y - np.log(w)-z-1)).mean()
    return dev


df_deviance = pd.DataFrame(
    index=["intercept", "glm", "local-glm-boost"],
    columns=["train", "test"],
    dtype=float,
)
df_deviance.loc["intercept", "train"] = deviance(y=y_train, z=z_opt, w=w_train).mean()
df_deviance.loc["intercept", "test"] = deviance(y=y_test, z=z_opt, w=w_test).mean()
df_deviance.loc["glm", "train"] = deviance(
    y=y_train, z=model.z0 + (model.beta0 * X_train).sum(axis=1), w=w_train
).mean()
df_deviance.loc["glm", "test"] = deviance(
    y=y_test, z=model.z0 + (model.beta0 * X_test).sum(axis=1), w=w_test
).mean()
df_deviance.loc["local-glm-boost", "train"] = deviance(
    y=y_train, z=model.predict(X_train), w=w_train
).mean()
df_deviance.loc["local-glm-boost", "test"] = deviance(
    y=y_test, z=model.predict(X_test), w=w_test
).mean()
df_deviance.to_csv(f"{output_path}results_deviance.csv")

# Negative log-likelihood
df_loss = pd.DataFrame(
    index=["intercept", "glm", "local-glm-boost"],
    columns=["train", "test"],
    dtype=float,
)
df_loss.loc["intercept", "train"] = model.distribution.loss(
    y=y_train, z=z_opt, w=w_train
).mean()
df_loss.loc["intercept", "test"] = model.distribution.loss(
    y=y_test, z=z_opt, w=w_test
).mean()
df_loss.loc["glm", "train"] = model.distribution.loss(
    y=y_train, z=model.z0 + (model.beta0 * X_train).sum(axis=1), w=w_train
).mean()
df_loss.loc["glm", "test"] = model.distribution.loss(
    y=y_test, z=model.z0 + (model.beta0 * X_test).sum(axis=1), w=w_test
).mean()
df_loss.loc["local-glm-boost", "train"] = model.distribution.loss(
    y=y_train, z=model.predict(X_train), w=w_train
).mean()
df_loss.loc["local-glm-boost", "test"] = model.distribution.loss(
    y=y_test, z=model.predict(X_test), w=w_test
).mean()
df_loss.to_csv(f"{output_path}results_loss.csv")

# Save the total CV losses
loss_train = pd.DataFrame(data=np.sum(loss["train"], axis=0), columns=features)
loss_valid = pd.DataFrame(data=np.sum(loss["valid"], axis=0), columns=features)
loss_train.to_csv(f"{output_path}loss_tuning_train.csv")
loss_valid.to_csv(f"{output_path}loss_tuning_valid.csv")


# Crate a dataframe with all models predictions on the validation data
predictions = pd.DataFrame(columns=df_loss.index, index=y_test.index)
predictions["y"] = y_test
predictions["w"] = w_test
predictions["intercept"] = np.exp(z_opt) * np.ones(len(y_test))
predictions["glm"] = np.exp(model.z0 + (model.beta0 * X_test).sum(axis=1))
predictions["local-glm-boost"] = np.exp(model.predict(X_test))
predictions.to_csv(f"{output_path}predictions.csv")

# Create a dataframe with feature importances
feature_importances = pd.DataFrame(index=features, columns=features)
for feature in features:
    if n_estimators[feature] != 0:
        feature_importances.loc[feature] = model.compute_feature_importances(
            feature, normalize=False
        )
feature_importances.to_csv(f"{output_path}feature_importances.csv")

df_n_estimators = pd.DataFrame(
    data=n_estimators.values(), index=n_estimators.keys(), columns=["n_estimators"]
)
df_n_estimators.to_csv(f"{output_path}n_estimators.csv")

beta_estimates = pd.DataFrame(index=features, columns=["beta0"])
beta_estimates.loc["intercept"] = z_opt
for j, feature in enumerate(features):
    beta_estimates.loc[feature] = model.beta0[j]
beta_estimates.to_csv(f"{output_path}beta_estimates.csv")

logger.log("Done!")
