import os

import numpy as np
import pandas as pd

from local_glm_boost.local_glm_boost import LocalGLMBooster
from local_glm_boost.tune_kappa import tune_kappa
from local_glm_boost.logger import LocalGLMBoostLogger

script_dir = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(script_dir, "../data/results/simulation_study")
os.makedirs(output_path, exist_ok=True)

logger = LocalGLMBoostLogger(
    verbose=2,
    output_path=output_path,
)

logger.log("Simulating data...")
# Set up simulation metadata
n = 20000
p = 8
rng = np.random.default_rng(0)
cov = np.eye(p)
cov[1, 7] = cov[7, 1] = 0.5
X = rng.multivariate_normal(np.zeros(p), cov, size=n)

# Define feature attentions
betas = [[]] * p
betas[0] = 0.5 * np.ones(n)
betas[1] = -0.25 * X[:, 1]
betas[2] = 0.5 * np.abs(X[:, 2]) * np.sin(2 * X[:, 2]) / X[:, 2]
betas[3] = np.zeros(n)
betas[4] = 0.5 * X[:, 3]
betas[5] = (1 / 8) * X[:, 4] ** 2
betas[6] = np.zeros(n)
betas[7] = np.zeros(n)
beta = np.stack(betas, axis=1).T

# Simulate
z0 = 0
mu = z0 + np.sum(beta.T * X, axis=1)
y = rng.normal(mu, 1)

idx = np.arange(n)
rng.shuffle(idx)
idx_train, idx_test = idx[: int(0.5 * n)], idx[int(0.5 * n) :]
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

logger.log("Tuning kappa...")
max_depth = 2
min_samples_leaf = 10
distribution = "normal"
kappa_max = 3000
eps = 0.01
n_splits = 2

tuning_results = tune_kappa(
    X=X_train,
    y=y_train,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution=distribution,
    kappa_max=kappa_max,
    eps=eps,
    n_splits=n_splits,
    rng=rng,
    logger=logger,
)

kappa_opt = tuning_results["kappa"]

for j in range(p):
    logger.log(f"Optimal kappa for covariate {j}: {kappa_opt[j]}")

# Evaluate performance
logger.log("Fitting models...")
model = LocalGLMBooster(
    kappa=kappa_opt,
    eps=eps,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution="normal",
)
model.fit(X=X_train, y=y_train, glm_init=True)
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
    index=[j for j in range(p) if kappa_opt[j] > 0], columns=range(p), dtype=float
)
for j in feature_importance.index:
    feature_importance.loc[j] = model.feature_importances(j=j, normalize=True)

feature_importance.index = [f"beta_{j}" for j in feature_importance.index]
feature_importance.columns = [f"x_{j}" for j in feature_importance.columns]

# Save results
logger.log("Saving results...")
results.to_csv(f"{output_path}/MSE.csv")
for data_set in ["train", "valid"]:
    pd.DataFrame(tuning_results["loss"][data_set].sum(axis=0)).to_csv(
        f"{output_path}/tuning_loss_{data_set}.csv"
    )

pd.DataFrame(kappa_opt).to_csv(f"{output_path}/kappa_opt.csv")
pd.DataFrame(model.beta0).to_csv(f"{output_path}/beta0.csv")

feature_importance.to_csv(f"{output_path}/feature_importance.csv")

os.makedirs(f"{output_path}/mu_hat", exist_ok=True)
os.makedirs(f"{output_path}/beta_hat", exist_ok=True)
for model_name in mu_hat.keys():
    np.savetxt(f"{output_path}/mu_hat/{model_name}.csv", mu_hat[model_name])
for model_name in beta_hat.keys():
    beta_hat[model_name].to_csv(f"{output_path}/beta_hat/{model_name}.csv")

logger.log("Done!")
