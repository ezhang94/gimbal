import unittest
from context import distributions

import numpy as onp
import jax.numpy as jnp
import jax.random as jr

from copy import deepcopy

class TestVMFG2(unittest.TestCase):
    def setUp(self):
        """Intialize parameters for vMFG instance."""

        seed = iter(jr.split(jr.PRNGKey(329), 4))

        batch_shape = (4,3)
        event_shape = (2,)

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

        self.base_params = dict(
            mean_directions=mean_directions,
            concentrations=concentrations,
            radii=radii,
            variances=variances,
        )

        self.base_vmfg = \
            distributions.vonMisesFisherGaussian(*self.base_params.values())

    def tearDown(self):
        pass

    def test_batch_shape(self):
        self.assertEqual(self.base_vmfg.batch_shape, (4,3))

    def test_event_shape(self):
        self.assertEqual(self.base_vmfg.event_shape, (2,))

    def test_reinterpreted_batch_ndim(self):
        independent_vmfg = \
            distributions.vonMisesFisherGaussian(*self.base_params.values(),
                                                 reinterpreted_batch_ndim=1)
        self.assertEqual(independent_vmfg.event_shape, (3,2,))

    # def test_log_prob_1(self):
    #     """Assert that log probability of samples is greater under generating
    #     distribution than under distribution with HIGHER CONCENTRATION VALUES.
    #     """
    #     seed = jr.PRNGKey(553)
    #     sample_shape = (10,)
    #     samples = self.base_vmfg.sample(sample_shape, seed)

    #     temp_params = deepcopy(self.base_params)
    #     temp_params['concentrations'] = 5 * temp_params['concentrations']
    #     temp_vmfg = distributions.vonMisesFisherGaussian(*temp_params.values())
        
    #     base_lp = jnp.sum(self.base_vmfg.log_prob(samples))
    #     temp_lp = jnp.sum(temp_vmfg.log_prob(samples))
    #     print(base_lp, temp_lp)
    #     self.assertTrue(base_lp > temp_lp)

    # def test_log_prob_2(self):
    #     """Assert that log probability of samples is greater under generating
    #     distribution than under distribution with HIGHER VARIANCE VALUES.
    #     """
    #     pass
    
    # def test_log_prob_independent(self):
    #     """Assert that log probability of samples
    #     """
    #     pass

    # def test_log_prob_given_center(self):
    #     """Assert that log probability of samples
    #     """
    #     pass


# From command line, run:
#   python -m unittest -v test_distributions.py 
if __name__ == '__main__':
    unittest.main()