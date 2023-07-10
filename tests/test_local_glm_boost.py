import unittest

import numpy as np

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

    def test_normal_loss(
        self,
    ):
        """
        Test the loss results on a normal distribution
        """
        y = self.rng.normal(self.z, 1)
        model = LocalGLMBooster(
            distribution="normal",
            n_estimators=[50, 40],
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=2,
        )
        model.fit(X=self.X, y=y)

        self.assertAlmostEqual(
            model.distribution.loss(y=y, z=model.predict(X=self.X)).mean(),
            2.464465952769601,
            places=3,
        )

    def test_normal_tuning(self):
        """
        Test the tuning of the number of estimators
        """
        y = self.rng.normal(self.z, 1)
        tuning_results = tune_n_estimators(
            X=self.X,
            y=y,
            distribution="normal",
            learning_rate=0.1,
            n_estimators_max=50,
            rng=self.rng,
            n_splits=2,
            min_samples_leaf=20,
            max_depth=2,
        )
        n_estimators = tuning_results["n_estimators"]
        n_estimators_expected = [50, 40]
        for i, kappa in enumerate(n_estimators_expected):
            self.assertEqual(
                n_estimators_expected[i],
                n_estimators[i],
                msg=f"Optimal n_estimators for normal distribution not as expected for dimension {i}",
            )

    def test_normal_feature_importance(self):
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
