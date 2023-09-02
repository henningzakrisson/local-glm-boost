import unittest

import numpy as np
import pandas as pd

from local_glm_boost import LocalGLMBooster
from local_glm_boost.utils.tuning import tune_n_estimators


class LocalGLMBoosterTestCase(unittest.TestCase):
    """
    A class that defines unit tests for the LocalGLMBooster class.
    """

    def setUp(self):
        """
        Set up for the unit tests.
        """
        self.rng = np.random.default_rng(seed=11)
        n = 1000
        p = 2
        self.X = self.rng.normal(0, 1, (n, p))

        self.z0 = 0
        betas = [[]] * p
        betas[0] = -1 * self.X[:, 0]
        betas[1] = 2 * self.X[:, 1]
        self.beta = np.stack(betas, axis=1).T
        self.z = self.z0 + np.sum(self.beta.T * self.X, axis=1)
        self.w = self.rng.choice([0.5, 1, 2], size=self.X.shape[0])

    def test_normal_loss(
        self,
    ):
        """
        Test the loss results on a normal distribution
        """
        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            n_estimators=[50, 37],
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
        )
        model.fit(X=self.X, y=y)

        self.assertAlmostEqual(
            model.distribution.loss(y=y, z=model.predict(X=self.X)).mean(),
            1.4386291517990328,
            places=3,
        )

    def test_gamma_loss(
        self,
    ):
        """
        Test the loss results on a gamma distribution
        """
        y = self.rng.gamma(1, np.exp(0.1 * self.z))
        model = LocalGLMBooster(
            distribution="gamma",
            n_estimators=[78, 191],
            learning_rate=0.01,
            min_samples_leaf=5,
            max_depth=2,
        )
        model.fit(X=self.X, y=y)

        self.assertAlmostEqual(
            model.distribution.loss(y=y, z=model.predict(X=self.X)).mean(),
            1.0784736678399922,
            places=3,
        )

    def test_poisson_loss(
        self,
    ):
        """
        Test the loss results on a Poisson distribution
        """
        y = self.rng.poisson(np.exp(0.1 * self.z))
        model = LocalGLMBooster(
            distribution="poisson",
            n_estimators=[6, 24],
            learning_rate=0.1,
            min_samples_leaf=5,
            max_depth=2,
        )
        model.fit(X=self.X, y=y)

        self.assertAlmostEqual(
            model.distribution.loss(y=y, z=model.predict(X=self.X)).mean(),
            0.8739431358249856,
            places=3,
        )

    def test_normal_loss_with_weights(
        self,
    ):
        """
        Test the loss results on a normal distribution with duration weights
        """
        y = self.rng.normal(self.w * self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            n_estimators=[48, 12],
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
        )
        model.fit(X=self.X, y=y, w=self.w)

        self.assertAlmostEqual(
            model.distribution.loss(y=y, z=model.predict(X=self.X), w=self.w).mean(),
            4.282556425015984,
            places=3,
        )

    def test_gamma_loss_with_weights(
        self,
    ):
        """
        Test the loss results on a gamma distribution with duration weights
        """
        y = self.rng.gamma(self.w, np.exp(0.1 * self.z))
        model = LocalGLMBooster(
            distribution="gamma",
            n_estimators=[78, 191],
            learning_rate=0.01,
            min_samples_leaf=5,
            max_depth=2,
        )
        model.fit(X=self.X, y=y, w=self.w)

        self.assertAlmostEqual(
            model.distribution.loss(y=y, z=model.predict(X=self.X), w=self.w).mean(),
            1.281501654959411,
            places=3,
        )

    def test_poisson_loss_with_weights(
        self,
    ):
        """
        Test the loss results on a Poisson distribution with duration weights
        """
        y = self.rng.poisson(self.w * np.exp(0.1 * self.z))
        model = LocalGLMBooster(
            distribution="poisson",
            n_estimators=[6, 24],
            learning_rate=0.1,
            min_samples_leaf=5,
            max_depth=2,
        )
        model.fit(X=self.X, y=y, w=self.w)

        self.assertAlmostEqual(
            model.distribution.loss(y=y, z=model.predict(X=self.X), w=self.w).mean(),
            1.071401011919995,
            places=3,
        )

    def test_tuning(self):
        """
        Test the tuning of the number of estimators
        """
        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
        )
        tuning_results = tune_n_estimators(
            X=self.X,
            y=y,
            model=model,
            n_estimators_max=60,
            rng=self.rng,
            n_splits=2,
            parallel=True,
        )
        n_estimators = tuning_results["n_estimators"]
        n_estimators_expected = [55, 50]
        for i, kappa in enumerate(n_estimators_expected):
            self.assertEqual(
                n_estimators_expected[i],
                n_estimators[i],
                msg=f"Optimal n_estimators for normal distribution not as expected for dimension {i}",
            )

    def test_tuning_with_weights(self):
        """
        Test the tuning of the number of estimators
        """
        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
        )
        tuning_results = tune_n_estimators(
            X=self.X,
            y=y,
            w=self.w,
            model=model,
            n_estimators_max=50,
            rng=self.rng,
            n_splits=2,
            parallel=True,
        )
        n_estimators = tuning_results["n_estimators"]
        n_estimators_expected = [47, 50]
        for i, kappa in enumerate(n_estimators_expected):
            self.assertEqual(
                n_estimators_expected[i],
                n_estimators[i],
                msg=f"Optimal n_estimators for normal distribution not as expected for dimension {i}",
            )

    def test_feature_importance(self):
        """
        Test the feature importance calculation
        """
        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
            n_estimators=[19, 30],
        )
        model.fit(X=self.X, y=y)
        feature_importances = {
            j: model.compute_feature_importances(j) for j in range(self.X.shape[1])
        }
        expected_feature_importances = {0: [1, 0], 1: [0, 1]}
        for i in range(self.X.shape[1]):
            for j in range(self.X.shape[1]):
                self.assertAlmostEqual(
                    feature_importances[i][j],
                    expected_feature_importances[i][j],
                    places=1,
                    msg=f"Feature importance for attention {i} not as expected for feature {j}",
                )

    def test_glm_init(self):
        """
        Test the GLM initialization of the GLM
        """
        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
            n_estimators=[19, 30],
            glm_init=[True, False],
        )
        model.fit(X=self.X, y=y)
        self.assertNotEqual(model.beta0[0], 0, msg="GLM not initialized")
        self.assertEqual(model.beta0[1], 0, msg="GLM initialized when it shouldn't")

    def test_no_glm_init(self):
        """
        Test the initialization of the model when GLM initialization is not used
        """
        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            glm_init=False,
        )
        model.fit(X=self.X, y=y)
        for j in range(self.X.shape[1]):
            self.assertEqual(model.beta0[j], 0, msg="GLM initialized when it shouldn't")

    def test_pandas_support(self):
        """
        Test the support of pandas dataframes by making sure the prediction
        is invariant to column order
        """
        y = pd.Series(self.rng.normal(self.z, 1))
        X = pd.DataFrame(self.X, columns=["a", "b"])
        model = LocalGLMBooster(
            distribution="normal",
        )
        model.fit(X=X, y=y)
        self.assertEqual(
            model.distribution.loss(y=y, z=model.predict(X=X)).mean(),
            model.distribution.loss(y=y, z=model.predict(X=X[["b", "a"]])).mean(),
            msg="Prediction not invariant to column order",
        )

    def test_feature_selection(self):
        """
        Test the feature selection support
        """
        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal", feature_selection={0: [0], 1: [0]}
        )
        model.fit(X=self.X, y=y)
        feature_importances = {
            j: model.compute_feature_importances(j) for j in range(self.X.shape[1])
        }
        for i in range(self.X.shape[1]):
            self.assertEqual(
                feature_importances[i][1],
                0,
                msg=f"Feature importance for non-selected feature non-zero",
            )

    def test_feature_selection_pandas(self):
        """
        Test the feature selection support in a dataframe setting
        """
        y = pd.Series(self.rng.normal(self.z, 1))
        X = pd.DataFrame(self.X, columns=["a", "b"])
        model = LocalGLMBooster(
            distribution="normal", feature_selection={"a": ["a"], "b": ["a"]}
        )
        model.fit(X=X, y=y)
        feature_importances = {
            coefficient: model.compute_feature_importances(coefficient)
            for coefficient in X.columns
        }
        for coefficient in X.columns:
            self.assertEqual(
                feature_importances[coefficient]["b"],
                0,
                msg=f"Feature importance for non-selected feature non-zero",
            )

    def test_hyperparameter_setting(self):
        """Test so that the hyperparameters are set in the correct order etc."""
        y = pd.Series(self.rng.normal(self.z, 1))
        X = pd.DataFrame(self.X, columns=["a", "b"])
        n_estimators = [10, 20]
        min_samples_leaf = 10
        min_samples_split = {"b": 40, "a": 50}
        feature_selection = {"a": ["a"], "b": ["b"]}
        model = LocalGLMBooster(
            distribution="normal",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            feature_selection=feature_selection,
        )
        model.fit(X=X, y=y)

        for j, feature in enumerate(X.columns):
            self.assertEqual(
                model.n_estimators[j],
                n_estimators[j],
                msg=f"n_estimators not set correctly for feature {feature}",
            )
            self.assertEqual(
                model.min_samples_leaf[j],
                min_samples_leaf,
                msg=f"min_samples_leaf not set correctly for feature {feature}",
            )
            self.assertEqual(
                model.min_samples_split[j],
                min_samples_split[feature],
                msg=f"min_samples_split not set correctly for feature {feature}",
            )
            self.assertEqual(
                model.feature_selection[j][0],
                j,
                msg=f"feature_selection not set correctly for feature {feature}",
            )

    def test_parallel_training(
        self,
    ):
        """
        Test the loss results on a normal distribution
        """
        n = 1000
        p = 5
        X_cont = self.rng.normal(0, 1, size=(n, p - 3))
        X_cat = self.rng.choice(["a", "b", "c"], size=n)
        self.X = np.concatenate([X_cont, pd.get_dummies(X_cat)], axis=1)

        self.z0 = 0
        betas = [[]] * p
        betas[0] = -1 * self.X[:, 0]
        betas[1] = 2 * self.X[:, 1]
        betas[2] = 1 * self.X[:, 0]
        betas[3] = 3 * np.zeros(n)
        betas[4] = 0.1 * self.X[:, 0] * self.X[:, 1]
        self.beta = np.stack(betas, axis=1).T
        self.z = self.z0 + np.sum(self.beta.T * self.X, axis=1)

        y = self.rng.normal(self.z, 1)
        model_parallel = LocalGLMBooster(
            distribution="normal",
            n_estimators=50,
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
        )
        model_parallel.fit(X=self.X, y=y, parallel_fit=[[2, 3, 4]])

        model = LocalGLMBooster(
            distribution="normal",
            n_estimators=50,
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
        )
        model.fit(X=self.X, y=y)

        self.assertAlmostEqual(
            model.distribution.loss(y=y, z=model.predict(X=self.X)).mean(),
            model_parallel.distribution.loss(y=y, z=model.predict(X=self.X)).mean(),
            places=6,
        )

    def test_parallel_tuning(
        self,
    ):
        """
        Test the tuning results when fitting in parallel
        """
        n = 1000
        p = 5
        X_cont = self.rng.normal(0, 1, size=(n, p - 3))
        X_cat = self.rng.choice(["a", "b", "c"], size=n)
        self.X = np.concatenate([X_cont, pd.get_dummies(X_cat)], axis=1)

        self.z0 = 0
        betas = [[]] * p
        betas[0] = -1 * self.X[:, 0]
        betas[1] = 2 * self.X[:, 1]
        betas[2] = 1 * self.X[:, 0]
        betas[3] = 3 * np.zeros(n)
        betas[4] = 0.1 * self.X[:, 0] * self.X[:, 1]
        self.beta = np.stack(betas, axis=1).T
        self.z = self.z0 + np.sum(self.beta.T * self.X, axis=1)

        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            n_estimators=50,
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
        )

        tuning_rng = np.random.default_rng(123)
        tuning_results = tune_n_estimators(
            X=self.X,
            y=y,
            model=model,
            n_estimators_max=60,
            rng=tuning_rng,
            n_splits=2,
            parallel=True,
        )
        n_estimators = tuning_results["n_estimators"]

        tuning_rng = np.random.default_rng(123)
        tuning_results_parallel = tune_n_estimators(
            X=self.X,
            y=y,
            model=model,
            n_estimators_max=60,
            rng=tuning_rng,
            n_splits=2,
            parallel=True,
            parallel_fit=[[2, 3, 4]],
        )
        n_estimators_parallel = tuning_results_parallel["n_estimators"]

        for j in range(p):
            self.assertEqual(
                n_estimators[j],
                n_estimators_parallel[j],
                msg=f"n_estimators not set correctly for feature {j}",
            )
