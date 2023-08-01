import unittest
import time

import numpy as np
import pandas as pd

from local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators

rng = np.random.default_rng(seed=11)
n = 100000
p = 10
X = rng.normal(0, 1, (n, p))

z0 = 0
betas = [[]] * p
for j in range(p):
    betas[j] = X[:, 0]
beta = np.stack(betas, axis=1).T
z = z0 + np.sum(beta.T * X, axis=1)
w = np.ones_like(z)

y = rng.normal(w * z, 1)
model = LocalGLMBooster(
    distribution="normal",
    n_estimators=50,
    learning_rate=0.1,
    min_samples_leaf=20,
    max_depth=2,
)
start_time = time.time()
model.fit(X=X, y=y, w=w, cyclical = False)
stop_time = time.time()

print(f"Model fit time: {stop_time - start_time}")

loss = model.distribution.loss(
      y = y,
      z = model.predict(X=X),
      w = w,
).mean()
print(f"Model loss: {loss}")

for j in range(p):
    feature_importances = model.compute_feature_importances(feature = j)
    print(f"Feature {j} importance: {feature_importances}")