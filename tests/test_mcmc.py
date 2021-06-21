import unittest

from context import (DATA, PARAMS)
import util_io
import util

import mcmc

from jax import jit
import jax.numpy as jnp
import jax.random as jr

class TestMCMC(unittest.TestCase):
    def setUp(self):
        N = 100
        
        with jnp.load(DATA) as f:
            self.observations = jnp.asarray(f['observed_pos_2d'][:N])   # shape (N, C, 3, 4)
            self.init_positions = \
                            jnp.asarray(f['triangulated_pos_3d'][:N])   # shape (N, C, 3)
            self.positions = jnp.asarray(f['groundtruth_pos_3d'][:N])   # shape (N, K, 3)

        self.params = util_io.load_parameters(PARAMS)

        seed = jr.PRNGKey(123)
        self.seed, init_seed = jr.split(seed)
        self.samples = mcmc.initialize(init_seed,
                                       self.params, self.observations)

    def test_initialize(self):
        # Check that triangulated positions are close to DLC3D values
        dx = jnp.linalg.norm(self.samples['positions'] - self.init_positions,
                             axis=-1)
        mean_dx = jnp.nanmean(dx, axis=0) # shape (K,)
        
        err_tol = 1e-1
        self.assertTrue(jnp.all(mean_dx < err_tol),
                        msg=f"Expected mean triangulated per-keypoint error to be less than {err_tol}, but got {mean_dx}")

        # Check shapes of everything else
        N, C, K, D_obs = self.observations.shape
        self.assertTrue(self.samples['outliers'].shape==(N,C,K))
        self.assertTrue(self.samples['directions'].shape==(N,K,3))
        self.assertTrue(self.samples['heading'].shape==(N,))
        self.assertTrue(self.samples['pose_state'].shape==(N,))

        S = len(self.params['state_probability'])
        self.assertTrue(self.samples['transition_matrix'].shape==(S,S))
        
    def test_outliers(self):
        mocap2d = jnp.stack(
            [util.project(P, self.positions)
            for P in self.params['camera_matrices']], axis=1)

        outliers = mcmc.sample_outliers(self.seed, self.params,
                                        mocap2d, self.samples)
        
        # Projected MOCAP observations should not be outliers
        # except for MOCAP positions that were originally NaN
        isnan_mask = jnp.isnan(mocap2d[...,0])

        avg_prob_outlier = jnp.mean(outliers[~isnan_mask])

        self.assertTrue(avg_prob_outlier,
                        msg=f"Expected outliers[~isnan_mask] to be a finite value, got {avg_prob_outlier}).")

        self.assertTrue(avg_prob_outlier < 0.25,
                        msg=f"Expected projected ground truth data to be inliers on average, got probability is outlier = {avg_prob_outlier}.")
      
        self.assertTrue(jnp.all(outliers[isnan_mask]),
                        msg='Expected NaN observations to be marked as outliers.')

    def test_directions(self):

        directions = mcmc.sample_directions(self.seed, self.params,
                                            self.samples)

        self.assertTrue(jnp.all(directions[:,0,:] == 0),
                        msg=f'Expected all root node directions to be undefined (0), got \n{directions[:,0,:]}')

        # Sampled values should not vary significantly from one step to
        # the next, so the angular differences should be minimal.
        dtheta = util.signed_angular_difference(self.samples['directions'],
                                                directions,
                                                jnp.array([0,0,1.]))
        avg_angular_error = jnp.mean(jnp.abs(dtheta), axis=0)
        self.assertTrue(jnp.all(avg_angular_error[1:] < 1),
                        msg=f'Expected avg angular error < 1 deg, but got {avg_angular_error}')


# python -m unittest -v test_mcmc.py 
if __name__ == '__main__':
    unittest.main()