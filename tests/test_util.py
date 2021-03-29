import unittest
from context import util

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
        # Runs into overflow error (inf) when using jnp. Use onp here instead
        refr_val = onp.log(scipy.special.iv(nu, z)) 

        self.assertTrue(onp.isclose(test_val, refr_val, atol=1., rtol=0.))

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

# From command line, run:
#   python -m unittest -v test_util.py 
if __name__ == '__main__':
    unittest.main()