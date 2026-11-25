"""Run: python3 -m unittest discover -s tests -p 'test_hh_quartic*.py'."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / 'kinematics/fit_hh_quartic_xsec.py'
spec = importlib.util.spec_from_file_location('quartic_fit', path)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


class HHMonomialFit(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.linspace(-5, -0.2, 6), np.linspace(-30, 50, 5))
        self.points = np.column_stack((x.ravel(), y.ravel()))
        self.coeff = np.array([1.8, -1.1, 4e-5, 0.23, 0.003, 5e-5, -0.001, -3e-5, 8e-6])
        self.values = fit.design_matrix(*self.points.T) @ self.coeff
        self.errors = np.linspace(0.001, 0.01, len(self.values))

    def test_nine_physical_monomials(self):
        np.testing.assert_array_equal(fit.design_matrix(2., 3.), [1, 2, 3, 4, 6, 9, 12, 18, 36])

    def test_recovers_synthetic_coefficients_and_surface(self):
        coeff, _, predicted, report = fit.fit_polynomial(self.points, self.values, self.errors)
        np.testing.assert_allclose(coeff, self.coeff, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(predicted, self.values, rtol=1e-12)
        self.assertLess(report['max_abs_pull'], 1e-8)
        self.assertEqual(report['ndof'], len(self.values) - 9)

    def test_absolute_errors_are_not_rescaled_by_small_chi2(self):
        _, cov1, _, _ = fit.fit_polynomial(self.points, self.values, self.errors)
        _, cov2, _, _ = fit.fit_polynomial(self.points, self.values, 2 * self.errors)
        self.assertTrue(np.all(np.diag(cov1) > 0))
        np.testing.assert_allclose(cov2, 4 * cov1)

    def test_rank_deficient_grid_rejected(self):
        points = self.points.copy()
        points[:, 1] = 1
        with self.assertRaisesRegex(ValueError, 'constrain'):
            fit.fit_polynomial(points, self.values, self.errors)

    def test_invalid_errors_rejected(self):
        for error in [0, -1, np.nan]:
            errors = self.errors.copy()
            errors[0] = error
            with self.subTest(error=error), self.assertRaises(ValueError):
                fit.fit_polynomial(self.points, self.values, errors)

    def test_insufficient_points_rejected(self):
        with self.assertRaisesRegex(ValueError, 'more than nine'):
            fit.fit_polynomial(self.points[:9], self.values[:9], self.errors[:9])

    def test_interpolation_matches_data_and_masks_extrapolation(self):
        result = fit.interpolate(self.points, self.values, *self.points.T)
        np.testing.assert_allclose(result, self.values, rtol=1e-12)
        self.assertTrue(np.isnan(fit.interpolate(self.points, self.values, 1., 1.)))

    def test_interpolation_reproduces_an_affine_plane(self):
        values = 10 + self.points[:, 0] + 0.02 * self.points[:, 1]
        np.testing.assert_allclose(fit.interpolate(self.points, values, -2., 5.), 8.1)


if __name__ == '__main__':
    unittest.main()
