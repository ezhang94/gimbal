"""
Test modules of gimbal/mcmc.py

To run all tests:
  $ python -m unittest -v test_mcmc.py 
To run a single test:
  $ python test_mcmc.py <TestCaseClass>.<test_name>
"""

import unittest
import os
import h5py

from jax import jit
import jax.numpy as jnp
import jax.random as jr

from context import (DATA, PARAMS)
import util_io
import util

import mcmc
import run

class TestMCMC(unittest.TestCase):
    def setUp(self):
        N = 100
        
        with jnp.load(DATA) as f:
            self.observations = jnp.asarray(f['observed_pos_2d'][:N])   # shape (N, C, 3, 4)
            self.init_positions = \
                            jnp.asarray(f['triangulated_pos_3d'][:N])   # shape (N, C, 3)
            self.positions = jnp.asarray(f['groundtruth_pos_3d'][:N])   # shape (N, K, 3)

        self.params = util_io.load_parameters(PARAMS)
        self.params = mcmc.initialize_parameters(self.params)

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
        
    def test_lp(self):
        # Test log joint probability

        lp = mcmc.log_joint_probability(
                self.params, self.observations,
                self.samples['outliers'], self.samples['positions'], self.samples['directions'],
                self.samples['heading'], self.samples['pose_state'], self.samples['transition_matrix'],)

        self.assertTrue(jnp.isfinite(lp),
                        msg=f'Expected finite valued log probability, but got {lp}.')

    def test_positions(self):
        print("Initializing HMC sampler, this may take a while...")
        positions, kernel_results = \
            mcmc.sample_positions(self.seed, self.params,
                                  self.observations, self.samples,
                                  step_size=1e-1, num_leapfrog_steps=1)

        self.assertTrue(kernel_results.is_accepted,
                        msg=f'Expected proposed HMC samples to be accepted.')

        avg_gradient = jnp.mean(jnp.abs(
                            kernel_results.accepted_results
                                          .grads_target_log_prob[0]))

        self.assertTrue((avg_gradient > 1e-3) and (avg_gradient < 10),
                        msg=f'Expected average gradients to be on order 1e0, but got {avg_gradient}.')

        self.assertFalse(jnp.allclose(self.samples['positions'], positions))

        print('Confirm that test time takes ~200 s on CPU and ~60 s on GPU.')

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

        directions = mcmc.sample_directions(self.seed, self.params, self.samples)

        self.assertTrue(jnp.all(directions[:,0,:] == 0),
                        msg=f'Expected all root node directions to be undefined (0), got \n{directions[:,0,:]}')

        # Sampled values should not vary significantly from one step to
        # the next, so the angular differences should be minimal.
        dtheta = util.signed_angular_difference(self.samples['directions'],
                                                directions,
                                                jnp.array([0,0,1.]))
        avg_angular_error = jnp.mean(jnp.abs(dtheta), axis=0)
        self.assertTrue(jnp.all(avg_angular_error[1:] < 1),
                        msg=f'Expected avg angular error < 1 rad, but got {avg_angular_error}')

    def test_headings(self):
        headings = mcmc.sample_headings(self.seed, self.params, self.samples)

        # On average, heading change from frame-to-frame is small.
        # Sometimes it is large because of sudden moving, or when animal
        # is rearing and heading direction is very sensitive to the
        # SpineM - SpineF direction vector
        dh = headings[1:] - headings[:-1]
        dh = (dh + jnp.pi/2) % jnp.pi - jnp.pi/2
        
        avg_dh = jnp.mean(jnp.abs(dh))

        self.assertTrue(avg_dh < 1,
                        msg=f'Expected avg frame-to-frame heading difference < 1, but got {dh}')

    def test_state_and_transition(self):
        seed_0, seed_1 = jr.split(self.seed)

        old_state = self.samples['pose_state'].copy()
        old_matrix = self.samples['transition_matrix'].copy()

        self.samples['pose_state'] = \
            mcmc.sample_state(seed_0, self.params, self.samples)

        self.samples['transition_matrix'] = \
            mcmc.sample_transition_matrix(seed_1, self.params, self.samples)

        avg_state_match = jnp.mean(self.samples['pose_state'] == old_state)

        # Check that pose state changed
        self.assertTrue(avg_state_match < 0.2,
                        msg=f'Expected sample states to be significantly different from randomly initialized states, but got {avg_state_match*100}% agreement.')

        # Check that transition matrix changed
        self.assertFalse(jnp.all(old_matrix == self.samples['transition_matrix']),
                         msg='Do not expect randomly initialized transition matrix to match sampled matrix.')

class TestPredict(unittest.TestCase):
    def setUp(self):
        N = 100
        
        with jnp.load(DATA) as f:
            self.observations = jnp.asarray(f['observed_pos_2d'][:N])           # shape (N, C, 3, 4)
            self.init_positions = \
                            jnp.asarray(f['triangulated_pos_3d'][:N])           # shape (N, C, 3)
            self.positions = jnp.asarray(f['groundtruth_pos_3d'][:N])           # shape (N, K, 3)

        self.params = util_io.load_parameters(PARAMS)
        self.params = mcmc.initialize_parameters(self.params)
    
    def test_output_to_dict(self):
        seed = jr.PRNGKey(123)
        
        num_iter = 10 
        all_samples = \
                mcmc.predict(seed, self.params, self.observations, 
                            num_mcmc_iterations=num_iter)

        # Samples should all have length (num_iter,)
        self.assertTrue(all([len(v) == num_iter for v in all_samples.values()]),
                        msg=f"Expected all elements to have {num_iter} samples, but got {[len(v) for v in all_samples.values()]}")

        # Log probability should increase with iteration
        lps = all_samples['log_probability']
        self.assertTrue(jnp.mean((lps[1:] - lps[:-1]) > 0) > 0.6,
                        msg=f"Expected log probability to increase on average with iteration, but got log probabilities of \n    {lps}\n differences of \n    {lps[1:] - lps[:-1]}")

        self.assertTrue(all_samples['positions'].dtype == jnp.float32,
                        msg=f"Did not enable double precision. Expected float32 dtype, but got {all_samples['positions'].dtype} dtype.")

    def test_output_to_hdf5(self):
        seed = jr.PRNGKey(123)
        
        num_iter = 11
        out_path = \
                mcmc.predict(seed, self.params, self.observations, 
                            num_mcmc_iterations=num_iter,
                            enable_x64=True,
                            out_options={'path':'test.hdf5', 'chunk_size':2})

        with h5py.File(out_path) as f:
            sample_keys = list(f)
            
            # Samples should all have length (num_iter,)
            self.assertTrue(all([len(f[k]) == num_iter for k in sample_keys]),
                            msg=f"Expected all elements to have {num_iter} samples, but got {[len(f[k]) for k in sample_keys]}")

            # Log probability should increase with iteration
            lps = f['log_probability']
            self.assertTrue(jnp.mean((lps[1:] - lps[:-1]) > 0) > 0.6,
                        msg=f"Expected log probability to increase on average with iteration, but got log probabilities of \n    {lps}\n differences of \n    {lps[1:] - lps[:-1]}")
            
            self.assertTrue(f['positions'].dtype == jnp.float64,
                        msg=f"Enabled double precision. Expected float64 dtype, but got {f['positions'].dtype} dtype.")

    
    def tearDown(self):
        if os.path.isfile('test.hdf5'):
            print('Cleaning up...Removing `test.hdf5`')
            os.remove('test.hdf5')
        
if __name__ == '__main__':
    unittest.main()