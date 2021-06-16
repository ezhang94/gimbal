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
# Geometry
# =================================================================== 

def xyz_to_uv(x_3d, u_axis, v_axis):
    """Project 3D coordinates located in a 2D subspace defined by (u,v).

    Parameters:
        x_3d: ndarray, shape (...,3)
            Coordinates in 3D to project.
        u_axis: array, shape (3,)
        v_axis: array, shape (3,)
            Unit-norm vectors defining the 2D U-V plane.
            Vectors are orthogonal.

    Returns:
        x_2d: ndarray, shape (...,2)
    """

    return jnp.stack(
        [jnp.sum(x_3d * u_axis, axis=-1),  # New U-coordinate
         jnp.sum(x_3d * v_axis, axis=-1),  # New V-coordinate
        ], axis=-1)

def uv_to_xyz(x_2d, u_axis, v_axis):
    """Map 2D coordinates in (u,v) coordinate system into 3D space.

    Parameters:
        x_2d: ndarray, shape (...,2)
            Coordinates in 2D to map.
        u_axis: array, shape (3,)
        v_axis: array, shape (3,)
            Unit-norm vectors defining the 2D U-V plane.
            Vectors are orthogonal.

    Returns:
        x_3d: ndarray, shape (...,3)
    """

    return x_2d[...,[0]] * u_axis + x_2d[...,[1]] * v_axis

def cartesian_to_polar(us):
    """Transform Cartesian direction vectors to polar coordinates.

    Parameters
    ----------
        us: ndarray, shape (..., 3)
            Unit direction vectors

    Returns
    -------
        thetas: ndarray, shape (...,)
            Polar angles
        phis: ndarray, shape (...,)
            Azimuthal angles
    """
    # Polar angle = atan(y/x)
    thetas = jnp.arctan2(us[...,1], us[...,0])

    # Azimuthal angle = acos(z) = atan(sqrt(x**2 + y **2) / z)
    phis = jnp.arctan2(jnp.sqrt(us[...,0]**2 + us[...,1]**2), us[...,2])
    
    return thetas, phis

def R_mat(thetas):
    """Generate 3D rotation matrix given polar angles (x-y plane only).
    
    Parameters
    ----------
        thetas: (...,)
    
    Returns
    -------
        Rs: ndarray, shape (..., 3, 3)
    """
    batch_shape = jnp.atleast_1d(thetas).shape

    Rs = jnp.tile(jnp.eye(3), (*batch_shape, 1, 1))
    Rs = Rs.at[...,0,0].set(jnp.cos(thetas))
    Rs = Rs.at[...,1,1].set(jnp.cos(thetas))
    Rs = Rs.at[...,0,1].set(-jnp.sin(thetas))
    Rs = Rs.at[...,1,0].set(jnp.sin(thetas))
    
    return jnp.squeeze(Rs)

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
