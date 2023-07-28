import numpy as np
import pandas as pd
import ssl
import os
import yaml
import shutil

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators
from local_glm_boost.utils.logger import LocalGLMBoostLogger

config_name = "real_data_study"

# Set up output folder, configuration file, run_id and logger
script_dir = os.path.dirname(os.path.realpath(__file__))
folder_path = os.path.join(script_dir, "../data/results/")
config_path = os.path.join(script_dir, f"{config_name}.yaml")
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

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

output_path = os.path.join(folder_path, f"run_{run_id}")
os.makedirs(output_path)

# Put the config in the output folder
shutil.copyfile(config_path, f"{output_path}/config.yaml")

# Set up logger
logger = LocalGLMBoostLogger(
    verbose=2,
    output_path=output_path,
)
logger.append_format_level(f"run_{run_id}")

# Load stuff from the config
logger.log("Loading configuration")
n = config["n"]
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
ssl._create_default_https_context = ssl._create_unverified_context
df = fetch_openml(data_id=41214, as_frame=True, parser="pandas").data

df = df.loc[df["IDpol"] >= 24500]
df = df.loc[df["ClaimNb"] < 5]
df = df.loc[df["Exposure"] < 1]
df["Diesel"] = (df["VehGas"] == "'Diesel'").astype(float)
if n != "all":
    df = df.sample(n)
n = len(df)

X = df[features_to_use]
y = df[target]
w = df[weights]

X = X / X.var(axis=0)

# Tune n_estimators
logger.log("Tuning model")
rng = np.random.default_rng(seed=random_seed)
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=test_size, random_state=rng.integers(0, 10000)
)

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
model.fit(X_train, y_train, w_train)

# Light model
feature_importances = {
    coefficient: model.compute_feature_importances(j=coefficient)
    for coefficient in features_to_use
    if n_estimators[coefficient] > 0
}
features_light = {
    coefficient: [
        feature
        for feature in features_to_use
        if feature_importances[coefficient][feature] > 0.05
    ]
    for coefficient in feature_importances.keys()
}

glm_init_light = {
    coefficient: np.abs(model.beta0[i]) > 0.05
    for i, coefficient in enumerate(features_to_use)
}

model_light = LocalGLMBooster(
    distribution="poisson",
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    min_samples_leaf=min_samples_leaf,
    max_depth=max_depth,
    features=features_light,
    glm_init=glm_init_light,
)
model_light.fit(X=X_train, y=y_train, w=w_train)

# Intercept model
to_minimize = lambda z: model.distribution.loss(y=y_train, z=z, w=w_train).mean()
z0 = np.log((y_train / w_train).mean())
from scipy.optimize import minimize

res = minimize(
    to_minimize, z0, method="nelder-mead", options={"xatol": 1e-8, "disp": False}
)
z_opt = res.x

# Summarize results
logger.log("Summarizing results")


# Define the Poisson deviance
def deviance(y, z, w):
    y_log_y = np.zeros_like(y)
    y_log_y[y > 0] = y[y > 0] * np.log(y[y > 0])
    return 2 * (y_log_y - y * (np.log(w) + z + 1) + w * np.exp(z))


# Poisson deviance
df_loss = pd.DataFrame(
    index=["intercept", "glm", "local-glm-boost", "local-glm-boost-light"],
    columns=["train", "test"],
    dtype=float,
)
df_loss.loc["intercept", "train"] = deviance(y=y_train, z=z_opt, w=w_train).mean()
df_loss.loc["intercept", "test"] = deviance(y=y_test, z=z_opt, w=w_test).mean()
df_loss.loc["glm", "train"] = deviance(
    y=y_train, z=model.z0 + (model.beta0 * X_train).sum(axis=1), w=w_train
).mean()
df_loss.loc["glm", "test"] = deviance(
    y=y_test, z=model.z0 + (model.beta0 * X_test).sum(axis=1), w=w_test
).mean()
df_loss.loc["local-glm-boost", "train"] = deviance(
    y=y_train, z=model.predict(X_train), w=w_train
).mean()
df_loss.loc["local-glm-boost", "test"] = deviance(
    y=y_test, z=model.predict(X_test), w=w_test
).mean()
df_loss.loc["local-glm-boost-light", "train"] = deviance(
    y=y_train, z=model_light.predict(X_train), w=w_train
).mean()
df_loss.loc["local-glm-boost-light", "test"] = deviance(
    y=y_test, z=model_light.predict(X_test), w=w_test
).mean()

# Negative log-likelihood
df_loss_2 = pd.DataFrame(
    index=["intercept", "glm", "local-glm-boost", "local-glm-boost-light"],
    columns=["train", "test"],
    dtype=float,
)
df_loss_2.loc["intercept", "train"] = model.distribution.loss(
    y=y_train, z=z_opt, w=w_train
).mean()
df_loss_2.loc["intercept", "test"] = model.distribution.loss(
    y=y_test, z=z_opt, w=w_test
).mean()
df_loss_2.loc["glm", "train"] = model.distribution.loss(
    y=y_train, z=model.z0 + (model.beta0 * X_train).sum(axis=1), w=w_train
).mean()
df_loss_2.loc["glm", "test"] = model.distribution.loss(
    y=y_test, z=model.z0 + (model.beta0 * X_test).sum(axis=1), w=w_test
).mean()
df_loss_2.loc["local-glm-boost", "train"] = model.distribution.loss(
    y=y_train, z=model.predict(X_train), w=w_train
).mean()
df_loss_2.loc["local-glm-boost", "test"] = model.distribution.loss(
    y=y_test, z=model.predict(X_test), w=w_test
).mean()
df_loss_2.loc["local-glm-boost-light", "train"] = model.distribution.loss(
    y=y_train, z=model_light.predict(X_train), w=w_train
).mean()
df_loss_2.loc["local-glm-boost-light", "test"] = model.distribution.loss(
    y=y_test, z=model_light.predict(X_test), w=w_test
).mean()

# Create dataframes with the results
# Crate a large df with both loss measures
df_loss_all = pd.concat([df_loss, df_loss_2], axis=1, keys=["deviance", "loss"])

# Save the total CV losses
loss_train = pd.DataFrame(data=np.sum(loss["train"], axis=0), columns=features_to_use)
loss_valid = pd.DataFrame(data=np.sum(loss["valid"], axis=0), columns=features_to_use)

# Crate a dataframe with all models predictions on the validation data
mu_hat = pd.DataFrame(columns=df_loss.index, index=y_test.index)
mu_hat["true"] = y_test / w_test
mu_hat["intercept"] = np.exp(z_opt) * np.ones(len(y_test))
mu_hat["glm"] = np.exp(model.z0 + (model.beta0 * X_test).sum(axis=1))
mu_hat["local-glm-boost"] = np.exp(model.predict(X_test))
mu_hat["local-glm-boost-light"] = np.exp(model_light.predict(X_test))

# Create a dataframe with feature importances
feature_importances = pd.DataFrame(index=features_to_use, columns=features_to_use)
for feature in features_to_use:
    if n_estimators[feature] != 0:
        feature_importances[feature] = model.compute_feature_importances(feature)

# Make a dataframe with n_esimators and inital beta values
kappa_beta = pd.DataFrame(index=features_to_use, columns=["n_estimators", "beta0"])
kappa_beta["n_estimators"] = n_estimators
kappa_beta["beta0"] = model.beta0

# Save the results as csv files
df_loss_all.to_csv(f"{output_path}/df_loss_all.csv")
loss_train.to_csv(f"{output_path}/loss_train.csv")
loss_valid.to_csv(f"{output_path}/loss_valid.csv")
mu_hat.to_csv(f"{output_path}/mu_hat.csv")
feature_importances.to_csv(f"{output_path}/feature_importances.csv")
kappa_beta.to_csv(f"{output_path}/kappa_beta.csv")

logger.log("Done!")
