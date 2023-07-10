# Import stuff
import os
import yaml

import numpy as np
import pandas as pd


from local_glm_boost.local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators
from local_glm_boost.utils.logger import LocalGLMBoostLogger

# Set up output folder, configuration file, run_id and logger
script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_dir, "config.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
folder_path = os.path.join(script_dir, "../../data/results/")
os.makedirs(folder_path, exist_ok=True)
run_id = (
    max(
        [int(folder_name.split("_")[1]) for folder_name in os.listdir(folder_path)]
        + [0]
    )
    + 1
)
output_path = os.path.join(folder_path, f"run_{run_id}")
os.makedirs(output_path)

logger = LocalGLMBoostLogger(
    verbose=2,
    output_path=output_path,
)
logger.append_format_level(f"run_{run_id}")

logger.log("Simulating data...")

# Set up simulation metadata
n = config["n"]
p = config["p"]
rng = np.random.default_rng(config["random_state"])
cov = np.eye(p)
correlations = config["correlations"]
if correlations is not None:
    for feature_1, feature_2, correlation in correlations:
        cov[feature_1, feature_2] = correlation
        cov[feature_2, feature_1] = correlation
X = rng.multivariate_normal(np.zeros(p), cov, size=n)

# Evaluate beta functions on X
betas = [eval(beta_code) for beta_code in config["beta_functions"]]
beta = np.stack(betas, axis=1).T

# Simulate
z0 = config["z0"]
mu = z0 + np.sum(beta.T * X, axis=1)
y = rng.normal(mu, 1)

idx = np.arange(n)
rng.shuffle(idx)
idx_train, idx_test = (
    idx[: int((1 - config["test_size"]) * n)],
    idx[int(config["test_size"] * n) :],
)
X_train, y_train, mu_train, beta_train = (
    X[idx_train],
    y[idx_train],
    mu[idx_train],
    beta[:, idx_train],
)
X_test, y_test, mu_test, beta_test = (
    X[idx_test],
    y[idx_test],
    mu[idx_test],
    beta[:, idx_test],
)

logger.log("Tuning model...")
max_depth = config["max_depth"]
min_samples_leaf = config["min_samples_leaf"]
distribution = config["distribution"]
n_estimators_max = config["n_estimators_max"]
learning_rate = config["learning_rate"]
n_splits = config["n_splits"]
glm_init = config["glm_init"]

tuning_results = tune_n_estimators(
    X=X_train,
    y=y_train,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution=distribution,
    n_estimators_max=n_estimators_max,
    learning_rate=learning_rate,
    glm_init=glm_init,
    n_splits=n_splits,
    rng=rng,
    logger=logger,
)

n_estimators = tuning_results["n_estimators"]

for j in range(p):
    logger.log(f"Optimal n_estimators for covariate {j}: {n_estimators[j]}")

# Evaluate performance
logger.log("Fitting models...")
model = LocalGLMBooster(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution="normal",
    glm_init=glm_init,
)
model.fit(X=X_train, y=y_train)
beta_hat = model.predict_parameter(X_test)

logger.log("Making predictions...")

beta_hat = {"true": pd.DataFrame(beta_test), "local_glm_boost": pd.DataFrame(beta_hat)}

mu_hat = {
    "true": mu_test,
    "intercept": y_train.mean() * np.ones(len(y_test)),
    "glm": (model.z0 + model.beta0.reshape(p) @ X_test.T),
    "local_glm_boost": model.predict(X_test),
}

logger.log("Calculating MSE...")
results = pd.DataFrame(
    index=["true", "intercept", "glm", "local_glm_boost"],
    columns=["Training MSE", "Test MSE"],
    dtype=float,
)
results.loc["true"] = [
    np.mean((mu_train - y_train) ** 2),
    np.mean((mu_test - y_test) ** 2),
]
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

for model_name in mu_hat.keys():
    logger.log(f"{model_name} MSE: {results.loc[model_name, 'Test MSE']}")

logger.log("Calculating feature importance...")
feature_importance = pd.DataFrame(
    index=[j for j in range(p) if n_estimators[j] > 0], columns=range(p), dtype=float
)
for j in feature_importance.index:
    feature_importance.loc[j] = model.compute_feature_importances(j=j, normalize=True)

feature_importance.index = [f"beta_{j}" for j in feature_importance.index]
feature_importance.columns = [f"x_{j}" for j in feature_importance.columns]

# Save results
logger.log("Saving results...")
results.to_csv(f"{output_path}/MSE.csv")
for data_set in ["train", "valid"]:
    pd.DataFrame(tuning_results["loss"][data_set].sum(axis=0)).to_csv(
        f"{output_path}/tuning_loss_{data_set}.csv"
    )

pd.DataFrame(n_estimators).to_csv(f"{output_path}/n_estimators.csv")
pd.DataFrame(model.beta0).to_csv(f"{output_path}/beta0.csv")

feature_importance.to_csv(f"{output_path}/feature_importance.csv")

os.makedirs(f"{output_path}/mu_hat", exist_ok=True)
os.makedirs(f"{output_path}/beta_hat", exist_ok=True)
for model_name in mu_hat.keys():
    np.savetxt(f"{output_path}/mu_hat/{model_name}.csv", mu_hat[model_name])
for model_name in beta_hat.keys():
    beta_hat[model_name].to_csv(f"{output_path}/beta_hat/{model_name}.csv")

logger.log("Done!")
