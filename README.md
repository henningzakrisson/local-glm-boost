# local-glm-boost
Python package that implements the LocalGLMBoost algorithm for explainable machine learning regression, a version
of a cyclically boosted tree-based varying coefficient model.
For more on the model, see the [preprint](https://arxiv.org/abs/2401.05982).

## Installation
You can install the package from the GitHub repository by following these steps:

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/henningzakrisson/local-glm-boost.git
    ```
2. Create a virtual environment in the root directory of the repository:
    ```bash
    python3 -m venv venv
    ```
3. Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```
4. Install the package
    ```bash
    pip install -e .
    ```
## Usage example
````python
from local_glm_boost import LocalGLMBooster
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Simulate data from a VCM
n = 1000
p = 3
X = np.random.normal(size=(n, p))
mu = 0.5 * X[:, 0] + 0.5 * X[:, 1]**2 + np.sin(X[:, 2])*X[:, 2]
y = np.random.normal(mu, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit a GLM using a normal distribution
glm = sm.GLM(y_train, X_train, family=sm.families.Gaussian()).fit()

# Fit a LocalGLMBoosting model
lgb = LocalGLMBooster(distribution='normal',
                      n_estimators = [0,100,100],
                      learning_rate = 0.01,
                      max_depth = 2)
lgb.fit(X_train, y_train)

# In-sample loss
print('In-sample loss')
print('GLM: ', ((glm.predict(X_train) - y_train)**2).mean())
print('LocalGLMboost: ', ((lgb.predict(X_train) - y_train)**2).mean())

# Out-of-sample loss
print('Out-of-sample loss')
print('GLM: ', ((glm.predict(X_test) - y_test)**2).mean())
print('LocalGLMboost: ', ((lgb.predict(X_test) - y_test)**2).mean())
````

## Contact
If you have any questions, feel free to contact me [here](mailto:henning.zakrisson@gmail.com).

