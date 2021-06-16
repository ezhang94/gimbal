import numpy as onp
import jax.numpy as jnp

# =================================================================== 
# Computer vision
# =================================================================== 

def project(P, Xs):
    """Project 3D positions to 2D for a given camera projection matrix.

    Parameters
    ----------
        P: ndarray, shape (3,4)
            Camera projection matrix
        Xs: ndarray, shape (....,3)
            3D coordinates in ambient world space
        
    Returns
    -------
        ys: ndarray, shape (...,2)
            2D coordinates in image plane
    """

    Xs_h = jnp.concatenate([Xs, jnp.ones((*Xs.shape[:-1],1))], axis=-1)
    ys_h = Xs_h @ P.T
    return ys_h[...,:-1] / ys_h[...,[-1]]

def triangulate(P1, P2, y1, y2,):
    """Triangulate 3D positions from given 2D observations.

    Wrapper function for OpenCV's triangulation function.

    Parameters
    ----------
        P1: ndarray, (3, 4)
        P2: ndarray, (3, 4)
            Camera projection matrices of images 1 and 2, respectively
        y1: ndarray, (..., 2)
        y2: ndarray, (..., 2)
            2D image coordinates from images 1 and 2, respectively
        
    Returns
    -------
        Xs: ndarray, shape (...,3)
            3D coordinates in ambient space
    """
    raise NotImplementedError

    # Unable to get OpenCV to run on cluster.
    # TODO Implement DLT algorithm in JAX
    # See OpenCV source code for reference
    #   https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/triangulation.cpp

    from cv2 import triangulatePoints

    # Xs_h: ndarray, shape (-1, 4). Homogeneous triangulated points
    Xs_h = triangulatePoints(onp.asarray(P1),
                             onp.asarray(P2),
                             onp.asarray(y1).reshape(-1,2).T,
                             onp.asarray(y2).reshape(-1,2).T).T
    
    # Xs: ndarray, shape (-1, 3). Cartesian normalized triangulated points.
    Xs = Xs_h[...,:-1] / Xs_h[...,[-1]]

    batch_shape = y1.shape[:-1]
    return Xs.reshape(*batch_shape, 3)

def triangulate_multiview(Ps, ys, camera_pairs=[]):
    """Robust triangulation of 3D positions from multiple 2D observations.

    Computes direct linear triangulation for each pair of camera views
    using OpenCv's triangulatePoints, then returns the median position
    across all triangulations.

    Parameters
    ----------
        Ps: ndarray, (C, 3, 4)
            Camera projection matrices
        ys: ndarray, (C, ..., 2)
            2D image coordinates
        camera_pairs: list of tuples, optional.
            Pairs of cameras to triangulate. If None (default), all
            possible pairs of cameras are triangulated.

    Returns
    -------
        Xs: ndarray, shape (...,3)
            3D coordinates in ambient space
    """
    
    C = len(Ps)
    batch_shape = ys.shape[1:-1]
    
    if not camera_pairs:
        camera_pairs = [(i,j) for i in range(C) for j in range(i+1, C)]

    # Triangulate
    Xs = jnp.empty((len(camera_pairs), *batch_shape, 3))
    for i, (c0, c1) in enumerate(camera_pairs):
        Xs = Xs.at[i].set(triangulate(Ps[c0], Ps[c1], ys[c0], ys[c1]))
    
    return Xs

# =================================================================== 
# Gaussians with constrained precision matrices
# =================================================================== 

def tree_graph_laplacian(parents, weights):
    """Generate weighted Laplcian matrix associated with a tree graph.

    Parameters
    ----------
        parents: array-like, length N
            parents[j] is the parent index of node j
            The parent of the root node is itself
        weights: nadarray, shape (N, D, D)

    Returns
    -------
        G: ndarray, shape (ND, ND)
    """

    N, D = len(parents), weights.shape[-1]
   
    G = jnp.zeros((N * D, N * D))
    for i in range(N): # Node i
        # Add self-iteration term
        G = G.at[i*D : (i+1)*D, i*D : (i+1)*D].add(weights[i])
        
        for j in range(i+1, N): # Children of node i
            if parents[j] == i:
                # Add degree term
                G = G.at[i*D:(i+1)*D, i*D:(i+1)*D].add(weights[j])

                # Subtract adjacency term
                G = G.at[i*D : (i+1)*D, j*D : (j+1)*D].add(-weights[j])
                G = G.at[j*D : (j+1)*D, i*D : (i+1)*D].add(-weights[j])
    return G

# =================================================================== 
# Safe math
# =================================================================== 
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
