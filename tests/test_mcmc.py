import unittest

from context import (DATA, PARAMS)
import util_io
import util

import mcmc

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

    def test_initialize(self):
        seed = jr.PRNGKey(123)
        samples = mcmc.initialize(seed, self.params, self.observations)

        # Check that positions are close to DLC3D values
        dx = jnp.linalg.norm(samples['positions'] - self.init_positions,
                             axis=-1)
        mean_dx = jnp.nanmean(dx, axis=0) # shape (K,)
        
        err_tol = 1e-1
        self.assertTrue(jnp.all(mean_dx < err_tol),
                        msg=f"Expected mean triangulated per-keypoint error to be less than {err_tol}, but got {mean_dx}")

        # Check shapes of everything else
        N, C, K, D_obs = self.observations.shape
        self.assertTrue(samples['outliers'].shape==(N,C,K))
        self.assertTrue(samples['directions'].shape==(N,K,3))
        self.assertTrue(samples['heading'].shape==(N,))
        self.assertTrue(samples['pose_state'].shape==(N,))

        S = len(self.params['state_probability'])
        self.assertTrue(samples['transition_matrix'].shape==(S,S))
        
    def test_outliers(self):
        mocap2d = jnp.stack(
            [util.project(P, self.positions)
            for P in self.params['camera_matrices']], axis=1)

        seed = jr.PRNGKey(123)
        outliers = mcmc.sample_outliers(seed, self.params, mocap2d,
                                        {'positions': self.positions})

        
        # Project MOCAP observations should not be outliers
        # except for MOCAP positions that were originally NaN
        isnan_mask = jnp.isnan(mocap2d[...,0])

        avg_prob_outlier = jnp.mean(outliers[~isnan_mask])

        self.assertTrue(avg_prob_outlier,
                        msg=f"Expected outliers[~isnan_mask] to be a finite value, got {avg_prob_outlier}).")

        self.assertTrue(avg_prob_outlier < 0.25,
                        msg=f"Expected projected ground truth data to be inliers on average, got probability is outlier = {avg_prob_outlier}.")
      
        self.assertTrue(jnp.all(outliers[isnan_mask]),
                        msg='Expected NaN observations to be marked as outliers.')
        

#   python -m unittest -v test_mcmc.py 
if __name__ == '__main__':
    unittest.main()