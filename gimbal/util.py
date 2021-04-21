import jax.numpy as jnp

def log_bessel_iv_asymptotic(x):
    """Logarithm of the asymptotic value of the modified Bessel function of
    the first kind :math:`I_nu`, for any order :math:`nu`.

    The asymptotic representation is given by Equation B.49 of the source as

    .. math:
        I_\nu(x) = \frac{1}{\sqrt{2\pi}} x^{-1/2} e^{x}

    for :math:`x\rightarrow\infty` with :math:`|\arg(x)| < \pi/2`.

    Source:
        Mainardi, F. "Appendix B: The Bessel Functions" in
        "Fractional Calculus and Waves in Linear Viscoelasticity,"
        World Scientific, 2010.
    """
    return x - 0.5 * (jnp.log(2 * jnp.pi * x))

def log_sinh(x):
    """Calculate the log of the hyperbolic sine function in a numerically stable manner.
    
    The sinh function is defined as

    .. math:
        \sinh(x) = \frac{e^x - e^{-x}}{2} = \frac{e^x * (1-e^{-2x}}{2}

    which yields the equation

    .. math:
        \log \sinh(x) = x + \log (1-e^{-2x}) - \log 2
    """
    return x + jnp.log(1 - jnp.exp(-2*x)) - jnp.log(2)

def coth(x):
    """Calculate the hyperbolic cotangent function and catch non-finite cases.

    The hyperbolic cotangent, which is the inverse of the hyperbolic tangent,
    is defined as 
    .. math:
         \coth(x) = \frac{e^{2x} + 1}{e^{2x} - 1}
    
    The asymptotic values of the hyperbolic cotangent, are given by

    .. math:
         \lim_{x \rightarrow \infty} \coth(x) = 1

    and

    .. math:
         \lim_{x \rightarrow 0} \coth(x) = 1/x = \infty
    """

    out = (jnp.exp(2*x) + 1) / (jnp.exp(2*x) - 1)
    
    # Replace nan values (which occur when dividing inf by inf) with 1's
    # Replace posinf values with large finite number (via posinf=None)
    return jnp.nan_to_num(out, nan=1.0, posinf=None, neginf=-jnp.inf)
