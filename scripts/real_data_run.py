config_name = "real_data_study"

# Import stuff
import yaml
import os
import shutil
import ssl

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from local_glm_boost.local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators
from local_glm_boost.utils.logger import LocalGLMBoostLogger

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

# Set up logger
logger = LocalGLMBoostLogger(
    verbose=2,
    output_path=output_path,
)
logger.append_format_level(f"run_{run_id}")


# Load data
logger.log("Loading data")
rng = np.random.default_rng(config["random_state"])
ssl._create_default_https_context = ssl._create_unverified_context
df = fetch_openml(data_id=41214, as_frame=True, parser="pandas").data
df = df.loc[df["IDpol"] >= 24500]
df = df.loc[df["ClaimNb"] < 5]
df = df.loc[df["Exposure"] < 1]
if config["n"] is not None:
    df = df.sample(config["n"], random_state=rng)
n = len(df)
feature_list = config["feature_list"]
p = len(feature_list)

X = df[feature_list]
X = X / X.var(axis=0)
y = df[config["target"]]
w = df[config["weights"]]

# Train test split
random_state = rng.integers(0, 2**32 - 1)
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=config["test_size"], random_state=random_state
)

# Tune n_estimators
logger.log("Tuning model")
max_depth = config["max_depth"]
min_samples_leaf = config["min_samples_leaf"]
distribution = config["distribution"]
n_estimators_max = config["n_estimators_max"]
learning_rate = config["learning_rate"]
n_splits = config["n_splits"]
glm_init = config["glm_init"]
features = config["features"]

model = LocalGLMBooster(
    n_estimators=0,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution=distribution,
    glm_init=glm_init,
    features=features,
)
tuning_results = tune_n_estimators(
    X=X_train,
    y=y_train,
    model=model,
    n_estimators_max=n_estimators_max,
    n_splits=n_splits,
    rng=rng,
    logger=logger,
    parallel=config["parallel"],
    n_jobs=config["n_jobs"],
    stratified=config["stratified"],
)
n_estimators = tuning_results["n_estimators"]

# Evaluate performance
logger.log("Fitting models")
model = LocalGLMBooster(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution="normal",
    glm_init=glm_init,
    features=features,
)
model.fit(X=X_train, y=y_train)
feature_importance = pd.DataFrame(
    index=[j for j in range(p) if n_estimators[j] > 0], columns=range(p), dtype=float
)
for j in feature_importance.index:
    feature_importance.loc[j] = model.compute_feature_importances(j=j, normalize=True)

# Also create a light-weight model
n_estimators_light = [
    n_estimator if n_estimator > 10 else 0 for n_estimator in n_estimators
]
glm_init_light = [np.abs(beta0) >= 0.05 for beta0 in model.beta0]
features_light = {
    coefficient: [
        feature
        for feature in feature_importance.loc[coefficient].index
        if feature_importance.loc[coefficient][feature] >= 0.05
    ]
    for coefficient in [
        coefficient
        for coefficient in range(0, p)
        if n_estimators_light[coefficient] > 0
    ]
}

model_light = LocalGLMBooster(
    n_estimators=n_estimators_light,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution="normal",
    glm_init=glm_init_light,
    features=features_light,
)

model_light.fit(X=X_train.values, y=y_train.values)

feature_importance_light = pd.DataFrame(
    index=[j for j in range(p) if n_estimators_light[j] > 0],
    columns=range(p),
    dtype=float,
)
for j in feature_importance_light.index:
    feature_importance_light.loc[j] = model_light.compute_feature_importances(
        j=j, normalize=True
    )

mu_hat = pd.DataFrame(
    columns=["intercept", "glm", "local_glm_boost", "local_glm_boost_light"],
)
mu_hat["intercept"] = y_train.mean() * np.ones(len(y_test))
mu_hat["glm"] = model.z0 + model.beta0.reshape(p) @ X_test.T
mu_hat["local_glm_boost"] = model.predict(X_test)
mu_hat["local_glm_boost_light"] = model_light.predict(X_test)

results = pd.DataFrame(
    index=["intercept", "glm", "local_glm_boost", "local_glm_boost_light"],
    columns=["Training MSE", "Test MSE"],
    dtype=float,
)
results.loc["intercept"] = [
    np.mean((y_train.mean() - y_train) ** 2),
    np.mean((y_train.mean() - y_test) ** 2),
]
results.loc["glm"] = [
    np.mean((model.z0 + model.beta0.reshape(p) @ X_train.T - y_train) ** 2),
    np.mean((model.z0 + model.beta0.reshape(p) @ X_test.T - y_test) ** 2),
]
results.loc["local_glm_boost"] = [
    np.mean((model.predict(X_train) - y_train) ** 2),
    np.mean((model.predict(X_test) - y_test) ** 2),
]

results.loc["local_glm_boost_light"] = [
    np.mean((model_light.predict(X_train) - y_train) ** 2),
    np.mean((model_light.predict(X_test) - y_test) ** 2),
]

# Save results
shutil.copyfile(config_path, f"{output_path}/config.yaml")

logger.log("Saving results...")
results.to_csv(f"{output_path}/MSE.csv")
for data_set in ["train", "valid"]:
    pd.DataFrame(np.sum(tuning_results["loss"][data_set], axis=0)).to_csv(
        f"{output_path}/tuning_loss_{data_set}.csv"
    )

pd.DataFrame(n_estimators).to_csv(f"{output_path}/n_estimators.csv")
beta0 = pd.DataFrame(columns=["local_glm_boost", "local_glm_boost_light"])
beta0["local_glm_boost"] = model.beta0
beta0["local_glm_boost_light"] = model_light.beta0
beta0.to_csv(f"{output_path}/beta0.csv")

feature_importance.index = [f"beta_{j}" for j in feature_importance.index]
feature_importance.columns = [f"x_{j}" for j in feature_importance.columns]
feature_importance.to_csv(f"{output_path}/feature_importance.csv")

feature_importance_light.index = [f"beta_{j}" for j in feature_importance_light.index]
feature_importance_light.columns = [f"x_{j}" for j in feature_importance_light.columns]
feature_importance_light.to_csv(f"{output_path}/feature_importance_light.csv")

mu_hat.to_csv(f"{output_path}/mu_hat.csv")

os.makedirs(f"{output_path}/beta_hat", exist_ok=True)
for model_name in beta_hat.keys():
    beta_hat[model_name].to_csv(f"{output_path}/beta_hat/{model_name}.csv")

logger.log_finish()
