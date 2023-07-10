import numpy as np
from local_glm_boost import LocalGLMBooster

n = 2000
p = 3
rng = np.random.default_rng(0)
cov = np.eye(p)
# cov[1, 7] = cov[7, 1] = 0.5
X = rng.multivariate_normal(np.zeros(p), cov, size=n)
z0 = 0

betas = [[]] * p
betas[0] = 0.5 * np.ones(n)
betas[1] = -0.5 * X[:, 1]
betas[2] = np.sin(2 * X[:, 0])
# betas[3] = 0.5 * X[:, 4]
# betas[4] = (1 / 8) * X[:, 5] ** 2
# betas[5] = np.zeros(n)
# betas[6] = np.zeros(n)
# betas[7] = np.zeros(n)
beta = np.stack(betas, axis=1).T

mu = z0 + np.sum(beta.T * X, axis=1)
y = rng.normal(mu, 1)

idx = np.arange(n)
rng.shuffle(idx)
idx_train, idx_test = idx[: int(0.5 * n)], idx[int(0.5 * n) :]
X_train, y_train, mu_train = X[idx_train], y[idx_train], mu[idx_train]
X_test, y_test, mu_test = X[idx_test], y[idx_test], mu[idx_test]

max_depth = 2
min_samples_leaf = 10
distribution = "normal"
kappa_max = 100
learning_rate = 0.1

n_estimators = [10] * p

model = LocalGLMBooster(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    distribution="normal",
    glm_init=True,
)
model.fit(X, y)

print(f"True MSE: {np.mean((y_test-mu_test)**2)}")
print(f"Intercept MSE: {np.mean((y_test-y_train.mean())**2)}")
print(f"GLM MSE: {np.mean((y_test-model.z0 - model.beta0.T @ X_test.T)**2)}")
print(f"Model MSE: {np.mean((y_test-model.predict(X_test))**2)}")

feature_importances = [
    model.compute_feature_importances(j=j, normalize=True) for j in range(p)
]
for j in range(p):
    for k in range(p):
        print(
            f"Feature importance for covariate {k} on beta_{j}: {feature_importances[k][j]}"
        )
