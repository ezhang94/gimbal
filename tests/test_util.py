import unittest
from context import (util, DATA)

import scipy.special
import numpy as onp
import jax.numpy as jnp

class TestMath(unittest.TestCase):
    def test_log_bessel_iv_asymptotic(self):
        nu = 5.
        z = 500.

        # This is an approximation, and we can't push z to be
        # too large before scipy.special.iv inf's out. So,
        # just ensure that test value is within 1 of reference value
        test_val = util.log_bessel_iv_asymptotic(z)

        # Runs into overflow error (inf) when using jnp (default float32)
        # Use onp here instead (default flaot64)
        refr_val = onp.log(scipy.special.iv(nu, z)) 

        self.assertTrue(onp.isclose(test_val, refr_val, atol=1., rtol=0.),
                        msg=f"Expected {refr_val}, got {test_val}.")

    def test_log_sinh(self):
        x = 20.

        # This is an exact expression, so we want to ensure that
        # values are very close to each other
        test_val = util.log_sinh(x)
        refr_val = jnp.log(jnp.sinh(x))

        self.assertEqual(test_val, refr_val)

    def test_coth(self):
        x = 50.

        test_val = util.coth(x)
        refr_val = 1/jnp.tanh(x)

        self.assertEqual(test_val, refr_val)

    def test_coth_asymptotic(self):
        x = jnp.array([1e-32, 1e32])

        test_val = util.coth(x)
        refr_val = jnp.array([jnp.nan_to_num(jnp.inf), 1.])

        self.assertTrue(jnp.all(test_val == refr_val))

class TestMultiview(unittest.TestCase):
    def setUp(self):
        with onp.load(DATA) as f:
            self.camera_matrices = jnp.asarray(
                    f['camera_matrices'])                               # shape (C, 3, 4)
            self.positions = jnp.asarray(
                    f['groundtruth_pos_3d'][:10])                       # shape (N, K, 3)

    def test_project_single_view(self):
        # observations have shape (c,n,k,2), with some NaNs.
        
        positions = self.positions[:, 2:7, :]                           # shape (N=2, K=5, 3)
        cmatrices = self.camera_matrices[-1]                            # shape (3, 4)
        observations = util.project(cmatrices, positions)               # shape (N, K, 2)

        N, K, _ = positions.shape

        self.assertEqual(observations.shape, (N, K, 2),
                        msg=f'Expected shape ({N},{K},2), got {observations.shape}.')

    def test_triangulate_singledim_nonan(self):
        # observations have shape (c,n,2), i.e. single-dim batch_shape
        # of (n,). No NaNs in data.

        positions = self.positions[:10, 0, :]                           # shape (N=10, 3)
        cmatrices = self.camera_matrices[:2]                            # shape (C=2, 3, 4)
        observations = jnp.stack(                                       # shape (C, N, 2)
            [util.project(P, positions) for P in cmatrices],
            axis=0)
        
        test_val = util.triangulate_dlt(cmatrices, observations)

        self.assertTrue(jnp.allclose(positions, test_val, atol=1e-2),
                        msg=f'Reference:\n{positions}\nTest:\n{test_val}')

    def test_triangulate_multidim_nonan(self):
        # observations have shape (c,n,k,2), i.e. multi-dim batch_shape
        # of (n,k). No NaNs

        positions = self.positions[:2, 0:5, :]                          # shape (N=2, K=5, 3)
        cmatrices = self.camera_matrices[:2]                            # shape (C=2, 3, 4)
        observations = jnp.stack(                                       # shape (C, N, K, 2)
            [util.project(P, positions) for P in cmatrices],
            axis=0)
        

        test_val = util.triangulate_dlt(cmatrices, observations)

        self.assertTrue(jnp.allclose(positions, test_val, atol=1e-2),
                        msg=f'Reference:\n{positions}\nTest:\n{test_val}')

    def test_triangulate_withnan(self):
        # observations have shape (c,n,k,2), with some NaNs.

        positions = self.positions[:2, 2:7, :]                          # shape (N=2, K=5, 3)
        cmatrices = self.camera_matrices[:2]                            # shape (C=2, 3, 4)
        observations = jnp.stack(                                       # shape (C, N, K, 2)
            [util.project(P, positions) for P in cmatrices],
            axis=0)

        test_val = util.triangulate_dlt(cmatrices, observations)

        mask = ~jnp.isnan(positions)
        finite_vals_allclose = jnp.allclose(positions[mask], test_val[mask], atol=1e-2)
        nan_vals_are_nan = jnp.all(jnp.isnan(test_val[~mask]))

        self.assertTrue(finite_vals_allclose,
                        msg=f'Finite values not close enough.\nReference:\n{positions}\nTest:\n{test_val}')
        self.assertTrue(nan_vals_are_nan,
                        msg=f'NaNs not propagated correctly.\nReference:\n{positions}\nTest:\n{test_val}')

    def test_triangulate_multiview_dlt(self):
        # Triangulate from C > 2 using DLT. If there are no outliers, then
        # results should be similar to when C=2, but perhaps more error.

        positions = self.positions[:, 0:2, :]                           # shape (N=10, K=2, 3)
        cmatrices = self.camera_matrices[:3]                            # shape (C=2, 3, 4)
        observations = jnp.stack(                                       # shape (C, N, K, 2)
            [util.project(P, positions) for P in cmatrices],
            axis=0)

        test_3 = util.triangulate_dlt(cmatrices, observations)
        test_2 = util.triangulate_dlt(cmatrices[:2], observations[:2])

        err_3 = jnp.mean(jnp.linalg.norm(test_3 - positions, axis=-1))
        err_2 = jnp.mean(jnp.linalg.norm(test_2 - positions, axis=-1))

        self.assertTrue(jnp.allclose(positions, test_3, atol=1e-1),
                        msg=f'Reference:\n{positions}\nTest:\n{test_3}')
        self.assertTrue(err_3 > err_2,
                        msg=f'Expected mean error of triangulated positions from 3 views ({err_3})to be greater than 2 views ({err_2}).')


    def test_triangulate_multiview_robust(self):
        # Triangulate from C > 2 using robust method. Expect robust
        # method to have lower error than multi-view DLT method.

        positions = self.positions[:, 0:2, :]                           # shape (N=10, K=2, 3)
        cmatrices = self.camera_matrices[:3]                            # shape (C=2, 3, 4)
        observations = jnp.stack(                                       # shape (C, N, K, 2)
            [util.project(P, positions) for P in cmatrices],
            axis=0)

        test_dlt = util.triangulate_dlt(cmatrices, observations)
        test_med = util.triangulate(cmatrices, observations)

        err_dlt = jnp.mean(jnp.linalg.norm(test_dlt - positions, axis=-1))
        err_med = jnp.mean(jnp.linalg.norm(test_med - positions, axis=-1))

        self.assertTrue(jnp.allclose(positions, test_med, atol=1e-2),
                        msg=f'Reference:\n{positions}\nTest:\n{test_med}')
        self.assertTrue(err_dlt > err_med,
                        msg=f'Expected mean error of triangulated positions using DLT ({err_dlt})to be greater than robust method ({err_med}).')


# From command line, run:
#   python -m unittest -v test_util.py 
if __name__ == '__main__':
    unittest.main()