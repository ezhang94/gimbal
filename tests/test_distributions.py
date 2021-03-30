import unittest
from context import distributions

import numpy as onp
import jax.numpy as jnp
import jax.random as jr

from copy import deepcopy

def generate_vmfg_params(seed, batch_shape, dim):
    event_shape = (dim,)
    
    seed = iter(jr.split(seed, 4))

    mean_directions = jr.normal(next(seed), batch_shape + event_shape)
    mean_directions /= jnp.linalg.norm(mean_directions, axis=-1, keepdims=True)

    conc_gamma_shape_param = 4.
    conc_gamma_scale_param = 10.
    concentrations = jr.gamma(next(seed), conc_gamma_shape_param, batch_shape)
    concentrations *= conc_gamma_scale_param

    radii = jr.uniform(next(seed), batch_shape, minval=0, maxval=10.)

    var_gamma_shape_param = 1.
    var_gamma_scale_param = 1.
    variances = jr.gamma(next(seed), var_gamma_shape_param, batch_shape)
    variances *= var_gamma_scale_param

    return dict(
            mean_directions=mean_directions,
            concentrations=concentrations,
            radii=radii,
            variances=variances,
        )

class TestVMFGProperties(unittest.TestCase):
    def setUp(self):
        """Intialize parameters for vMFG instance."""

        seed = jr.PRNGKey(329)

        self.base_batch_shape = (4,3)
        self.base_params = generate_vmfg_params(seed, self.base_batch_shape, 2)
        self.base_distr = \
            distributions.vonMisesFisherGaussian(*self.base_params.values())
    
    def test_batch_shape(self):
        self.assertEqual(self.base_distr.batch_shape, self.base_batch_shape)

    def test_event_shape(self):
        self.assertEqual(self.base_distr.event_shape, (2,))

    def test_reinterpreted_batch_ndim_event_shape(self):
        independent_distr = \
            distributions.vonMisesFisherGaussian(*self.base_params.values(),
                                                 reinterpreted_batch_ndim=1)
        self.assertEqual(independent_distr.event_shape,
                         self.base_batch_shape[-1:]+(2,))

class TestVMFG2(unittest.TestCase):
    def setUp(self):
        self.dim = 2    

    def test_log_vmf_normalizer_uniform(self):
        seed = jr.PRNGKey(1909)
        params = generate_vmfg_params(seed, (), self.dim)
        params['concentrations'] = jnp.zeros_like(params['concentrations'])
        
        distr = distributions.vonMisesFisherGaussian(*params.values())
        self.assertEqual(distr.log_vmf_normalizer(distr.concentration), -jnp.log(2 * jnp.pi))

    def test_vmf_mean_high_concentration(self):
        seed = jr.PRNGKey(1909)
        params = generate_vmfg_params(seed, (), self.dim)
        params['concentrations'] = 500 * jnp.ones_like(params['concentrations'])
        
        distr = distributions.vonMisesFisherGaussian(*params.values())
        self.assertTrue(jnp.allclose(distr.vmf_mean(), params['mean_directions'], atol=1e-2))
    
    def test_log_prob_1(self):
        # Assert that log probability of samples is greater under the generating
        # distribution than under distribution with HIGHER CONCENTRATION VALUES.

        seed = jr.PRNGKey(553)

        base_params = generate_vmfg_params(seed, (), self.dim)
        base_distr = distributions.vonMisesFisherGaussian(*base_params.values())
        samples = base_distr.sample((10,), seed)

        temp_params = deepcopy(base_params)
        temp_params['concentrations'] = 10. * temp_params['concentrations']
        temp_distr = distributions.vonMisesFisherGaussian(*temp_params.values())
        
        base_lp = jnp.mean(base_distr.log_prob(samples))
        temp_lp = jnp.mean(temp_distr.log_prob(samples))

        self.assertTrue(base_lp > temp_lp)

    def test_log_prob_2(self):
        # Assert that log probability of samples is greater under the generating
        # distribution than under distribution with HIGHER VARIANCE VALUES.

        seed = jr.PRNGKey(553)

        base_params = generate_vmfg_params(seed, (), self.dim)
        base_distr = distributions.vonMisesFisherGaussian(*base_params.values())
        samples = base_distr.sample((10,), seed)

        temp_params = deepcopy(base_params)
        temp_params['variances'] = 25. * temp_params['variances']
        temp_distr = distributions.vonMisesFisherGaussian(*temp_params.values())
        
        base_lp = jnp.mean(base_distr.log_prob(samples))
        temp_lp = jnp.mean(temp_distr.log_prob(samples))

        self.assertTrue(base_lp > temp_lp)
    
    def test_log_prob_independent(self):
        # Assert that the log probability of samples from a distribution 
        # with reinterpreted_batch_ndim > 1 are calculated correctly
        seed = jr.PRNGKey(553)

        params = generate_vmfg_params(seed, (4,), self.dim)
        reg_distr = distributions.vonMisesFisherGaussian(*params.values())
        ind_distr = distributions.vonMisesFisherGaussian(*params.values(),
                                                         reinterpreted_batch_ndim=1)
        samples = reg_distr.sample((10,), seed)

        reg_lp_corrected = jnp.sum(reg_distr.log_prob(samples), axis=-1)
        ind_lp = ind_distr.log_prob(samples)
        
        self.assertTrue(jnp.all(reg_lp_corrected == ind_lp))

class TestVMFG3(TestVMFG2):
    # Reuse all tests from TestVMFG2, but with 3D vMFG
    def setUp(self):
        self.dim = 3

    def test_log_vmf_normalizer_uniform(self):
        seed = jr.PRNGKey(1909)
        params = generate_vmfg_params(seed, (), self.dim)
        params['concentrations'] = jnp.zeros_like(params['concentrations'])
        
        distr = distributions.vonMisesFisherGaussian(*params.values())
        self.assertEqual(distr.log_vmf_normalizer(distr.concentration), -jnp.log(4 * jnp.pi))


# From command line, run:
#   python -m unittest -v test_distributions.py 
if __name__ == '__main__':
    unittest.main()