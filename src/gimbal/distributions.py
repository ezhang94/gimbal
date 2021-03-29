import jax.numpy as jnp
import jax.scipy.special
import jax.random as jr
import tensorflow_probability as tfp

from util import log_bessel_iv_asymptotic, log_sinh, coth

CONCENTRATION_REGULARIZER = 1e-8
VARIANCE_REGULARIZER = 1e-8

def vmf_mean_2d(mean_direction, concentration):
    """Calculate the mean of a 2D vMF distribution.
    
    The mean is given by

    .. math:
        \mathbb{E}[p(u)] = \frac{I_1(\kappa)}{I_0(\kappa)} \nu

    where :math:`I_j` denotes the modified Bessel function of the first kind
    and order j.
    
    As :math:`\lim \kappa\rightarrow\infty`, the values of :math:`I_0(\kappa)`,
    :math:`I_1(\kappa)`, and in fact, :math:`I_\alpha(\kappa)` for any order
    :math:`\alpha`, approach the same value. Therefore, for large concentrations,

    .. math:
        \lim_{\kappa \rightarrow \infty} I_1(\kappa)/I_0(\kappa) = 1

    Parameters:
        mean_direction: ndarray, shape (..., 2)
        concentration: ndarray, shape (...,)
    
    Returns:
        mean: ndarray, shape (..., 2)
    """

    resultant_length = jax.scipy.special.i1(concentration) / jax.scipy.special.i0(concentration)
    resultant_length = jnp.nan_to_num(resultant_length, nan=1.0,
                                      posinf=jnp.inf, neginf=-jnp.inf)

    return resultant_length[..., None] * mean_direction

def vmf_mean_3d(mean_direction, concentration):
    """Calculate the mean of a 3D vMF distribution.
    
    The mean is given by

    .. math:
         \mathbb{E}[p(u)] = (\coth(\kappa) - \frac{1}{\kappa}) \nu

    Source:
        Hillen, T., Painter, K., Swan, A., and Murtha, A. 
        "Moments of von Mises and Fisher distributions and applications."
        Mathematical Biosciences and Engineering, 2017, 14(3):673-694.
        doi: 10.3934/mbe.2017038

    Parameters:
        mean_direction: ndarray, shape (..., 3)
        concentration: ndarray, shape (...,)
    
    Returns:
        mean: ndarray, shape (..., 3)
    """

    resultant_length = coth(concentration) - 1./concentration
    return resultant_length[...,None] * mean_direction

def log_vmf_normalizer_2d(mean_direction, concentration):
    raise NotImplementedError

def log_vmf_normalizer_3d(mean_direction, concentration):
    raise NotImplementedError

# ---------------------------------------------------------------------------

class VMFGFunctionFactory():
    def __init__(self):
        self._functions = {
            2: {},
            3: {}
        }
    
    def register_function(self, function_name, dim, function_object):
        self._functions[dim][function_name] = function_object
    
    def get_function(self, function_name, dim):
        """Return the specified function object given distribution dimension."""
        dim_specific_functions = self._functions.get(dim)
        if dim_specific_functions is None:
            raise ValueError(dim)

        function_object = dim_specific_functions.get(function_name)
        if function_name is None:
            raise ValueError(function_name)
        return function_object

vmfg_factory = VMFGFunctionFactory()
vmfg_factory.register_function("VMF_MEAN", 2, vmf_mean_2d)
vmfg_factory.register_function("VMF_MEAN", 3, vmf_mean_3d)
vmfg_factory.register_function("LOG_VMF_NORMALIZER", 2, log_vmf_normalizer_2d)
vmfg_factory.register_function("LOG_VMF_NORMALIZER", 3, log_vmf_normalizer_3d)

# ===========================================================================

class vonMisesFisherGaussian:
    def __init__(self, mean_direction=None, concentration=None,
                 radius=None, variance=None, center=None, 
                 reinterpreted_batch_ndim=None):
        """The von Mises-Fisher-Gaussian distribution.

        Parameters:
            mean_direction: ndarray, shape (B1,... Bn, D)
              A unit vector indicating the mode of the distribution, or the unit
              direction of the mean. NOTE: `D` is currently restricted to {2,3}
            concentration: ndarray, shape (B1,... Bn)
              The level of concentration of samples around the `mean_direction`.
            radius: ndarray, shape (B1,... Bn)
              Radius of the sphere on which the data is supported.
            variance: ndarray, shape (B1,... Bn)
              Variance of data points about the surface of the sphere.
            center: ndarray, shape (B1,... Bn, D)
              Center of the sphere on which data is supported. [default: origin]
            reinterpreted_batch_ndim: integer
              Specifies the number of batch dims to be absorbed into event dim.
              [default: 0] See Notes for more detail about this parameter.
        
        Notes:
            The `reinterpreted_batch_ndim` parameter is inspired the parameter
            of the same name in tfp.distributions.Independent, which allows for
            the representation of a collection of independent, non-identical
            distributions as a single random variable.
            Concretely, when reinterpreted_batch_ndim = 0, we have the default
            batch_shape = (B1,...Bn) and event_shape = (D,).
            If reinterpreted_batch_ndim = 1, then we have a distribution with
            batch_shape = (B1,...Bn-1) and event_shape = (Bn, D,). Now, we have
            a collection Bn independent vMFG distributions of dimension D.
            Practically, this parameter affects the number of rightmost dims
            over which we sum the base distribution's log_prob.
        
        Source:
            Mukhopadhyay, M., Li D., and Dunson, D.
            "Estimating densities with non-linear support by using
            Fisher-Gaussian kernels." Journal of the Royal Statistical Society:
            Series B (Statistical Methodology), 2020, 82(5), 1249-1271.
            doi: 10.1111/rssb.12390
        """

        self._mean_direction = jnp.asarray(mean_direction)
        self._concentration = jnp.broadcast_to(concentration, self._mean_direction.shape[:-1])
        self._radius = jnp.broadcast_to(radius, self._mean_direction.shape[:-1])
        self._variance = jnp.broadcast_to(variance, self._mean_direction.shape[:-1])
        self._center = jnp.zeros_like(self._mean_direction) if center is None \
                            else jnp.broadcast_to(center, self._mean_direction.shape)

        # Specify batch and event shapes
        self._dim = self._mean_direction.shape[-1]
        assert self._dim == 2 or self._dim == 3, \
            "Dimension not supported, expected dim={2,3}, received {}".format(self.dim)
        
        self._reinterpreted_batch_ndim = 0 if reinterpreted_batch_ndim is None \
                                            else reinterpreted_batch_ndim
        self._event_ndim = 1 + self._reinterpreted_batch_ndim
        self._batch_shape = self._mean_direction.shape[:-self._event_ndim]
        self._event_shape = self._mean_direction.shape[-self._event_ndim:]
    
    @property
    def mean_direction(self):
        return self._mean_direction

    @property
    def concentration(self):
        return self._concentration

    @property
    def radius(self):
        return self._radius
        
    @property
    def variance(self):
        return self._variance

    @property
    def center(self):
        return self._center

    @property
    def dtype(self):
        return self.mean_direction.dtype

    @property
    def batch_shape(self):
        return self._batch_shape
    
    @property
    def event_shape(self):
        return self._event_shape

    @property
    def event_ndim(self):
        return self._event_ndim

    @property
    def dim(self):
        return self.event_shape[-1]


    # ==========================================================================

    def sample(self, sample_shape, seed):
        """Sample from vMFG distribution.
        
        The generative sampling model is given by
        ```
                u ~ vMF(mean_direction, concentration)
            x | u ~ MVN(radius * u + center, variance * I)
        ```

        Parameters:
            sample_shape: tuple
            seed : jax.random.PRNGKey
        
        Returns:
            pos_samploes: ndarray, shape (*sample_shape, *batch_shape, *event_shape)
        """
        
        seed_1, seed_2 = jr.split(seed)

        # Each direction sample is drawn independently from parameterized vMF distribution
        vmf_samples = \
            tfp.distributions.VonMisesFisher(self.mean_direction, self.concentration).sample(sample_shape, seed_1)
        
        # Each position sample is is located at `radius * u + center`,
        # with diagonal covariance specified by `variance`
        pos_samples = jnp.sqrt(self.variance) * jr.normal(seed_2, shape=vmf_samples.shape, dtype=self.dtype)
        pos_samples += self.radius * vmf_samples + self.center  # Add mean

        return pos_samples

    # ==========================================================================

    def vmf_mean(self,):
        """Mean of vMF distribution. Specific calculation method is optimized for
        each dimension.
        """
        _vmf_mean = vmfg_factory.get_function("VMF_MEAN", self.dim)
        return _vmf_mean(self.mean_direction, self.concentration)

    def log_vmf_normalizer(self,):
        """Calculates the log normalization constant of the vMF distribution.
        Specific calculation method is dimension dependent
        """
        _log_vmf_normalizer = vmfg_factory.get_function("LOG_VMF_NORMALIZER", self.dim)
        return _log_vmf_normalizer(self.concentration)

    # ==========================================================================

    def log_prob(self, x, c=None):
        c = self.center if c is None else c
        delta = x - c
        return self._log_prob(delta)

    def _log_prob(self, delta):
        """Calculate log probability of center-subtracted samples under vMFG distribution

        Parameters:
            delta: ndarray, shape (...,B1,...Bn,E1,...D))
        
        Returns:
            log_p: ndarray, shape(...,B1,...Bn)

        Notes:
        The probabiity density function of the vMFG distribution is

        .. math:
            p(x; \nu, \kappa, c, \rho, \sigma^2)
            = \frac{C_d(\kappa)}{C_d(\Vert \kappa \nu - (x-c) \rho/\sigma^2 \Vert_2)}
            \exp{-\frac{1}{2\sigma^2}((x-c)^2 + \rho^2)}.
        """
        
        D = delta.shape[-1]

        # shape (...,B,E-1)
        conc_tilde = self.mean_direction * self.concentration[...,None]
        conc_tilde += delta * (self.radius/self.variance)[...,None]
        conc_tilde = jnp.linalg.norm(conc_tilde, axis=-1)

        log_p = self.log_vmf_normalizer(self.concentration)
        log_p -= self.log_vmf_normalizer(conc_tilde)
        log_p -= 0.5 * D * jnp.log(2*jnp.pi*self.variance)

        log_p -= 0.5 * (jnp.linalg.norm(delta, axis=-1)**2 + self.radius**2)/self.variance

        # Now add dummy axis to generalize to reinterpreted_batch_ndim >0 cases
        log_p = log_p[...,None]
        reduce_axes = tuple(-(1+onp.arange(self.event_ndim)))
        
        return jnp.sum(log_p, axis=reduce_axes)
    