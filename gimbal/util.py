import numpy as onp
import jax.numpy as jnp

# =================================================================== 
# Computer vision
# =================================================================== 

def project(P, Xs):
    """Project 3D positions to 2D image coordinates.

    Parameters
    ----------
        Ps: ndarray, shape (3,4)
            Camera projection matrix
        Xs: ndarray, shape (....,3)
            3D coordinates in ambient world space
        
    Returns
    -------
        ys: ndarray, shape (...,2)
            2D image coordinates of 3D positions.
    """
    Xs_h = jnp.concatenate([Xs, jnp.ones((*Xs.shape[:-1],1))], axis=-1)
    ys_h = Xs_h @ P.T
    return ys_h[...,:-1] / ys_h[...,[-1]]

def triangulate_dlt(Ps, ys):
    """Triangulate 3D position between two 2D correspondances using the direct
    linear transformation (DLT) method.

    If any 2D correspondance is missing (i.e. NaN), returns triangulated
    position as NaN value as well.

    TODO: Normalize input data (see HZ, p104. "4.4 Transformation 
    invariance and normalization.", particulary 4.4.4): For each image,
        1.  Translate all points such that collection's centroid is
            about the origin.
        2.  Scale all points (Cartesian/non-homogeneous) so average
            distance is sqrt(2).

    Reference: Hartley and Zimmerman, p312; [OpenCV implementation]
    (https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/triangulation.cpp)

    Parameters
    ----------
        Ps: ndarray, shape (C,3,4)
            Camera matrices
        ys: ndarray, shape (C,...,2)
            2D image correspondances

    Returns
    -------
        x: ndarray, shape (...,3)

    """
    Ps, ys = jnp.atleast_3d(Ps), jnp.atleast_3d(ys)
    num_cameras = len(Ps)
    batch_shape = ys.shape[1:-1]

    A = jnp.empty((*batch_shape, 2*num_cameras,4))

    # Eliminate homogeneous scale factor via cross product
    for c in range(num_cameras):
        P, y = Ps[c], ys[c]
        A = A.at[..., 2*c,  :].set(y[..., [0]] * P[2] - P[0])
        A = A.at[..., 2*c+1,:].set(y[..., [1]] * P[2] - P[1])

    # Solution which minimizes algebraic corespondance error is
    # the right eigenvector associated with smallest eigenvalue.
    # Vh: ndarray, shape (N,4,4). Matrix of right eigenvectors
    _, _, Vh = jnp.linalg.svd(A)

    X = Vh[...,-1,:] # 3D position in homogeneous coordinates, shape (N,4)
    X = X[...,:-1] / X[...,[-1]]

    # Inputs with NaN entries produce NaN matrix blocks in A.
    # When SVD performed on these matrix blocks, returns matrix of -I
    # Resulting X vector = [0,0,0.], which is unidentifiable from a point
    # truly at the origin. Mask these points out and reset as NaN.
    # NBL: Appears unneeded with jax.numpy, but needed for regular numpy
    # isnan_mask = jnp.any(jnp.isnan(A), axis=(-1,-2)) # shape (...,)
    # X = X.at[isnan_mask].set(jnp.nan)

    return jnp.squeeze(X)


def triangulate(Ps, ys, camera_pairs=[]):
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
        Xs = Xs.at[i].set(triangulate_dlt(Ps[(c0,c1),:,:], ys[(c0,c1),...]))
    
    return jnp.median(Xs, axis=0)



def opencv_triangulate(Ps, ys, camera_pairs=[]):
    C = len(Ps)
    batch_shape = ys.shape[1:-1]
    if not camera_pairs:
        camera_pairs = [(i,j) for i in range(C) for j in range(i+1, C)]
    Xs = jnp.empty((len(camera_pairs), *batch_shape, 3))
    for i, (c0, c1) in enumerate(camera_pairs):
        Xs = Xs.at[i].set(opencv_triangulate_dlt(Ps[(c0,c1),:,:], ys[(c0,c1),...]))
    return jnp.median(Xs, axis=0)


def opencv_triangulate_dlt(Ps, ys):
    import numpy as np, cv2
    batch_shape = ys.shape[1:-1]
    Ps = np.array(Ps)
    ys = np.array(ys).reshape(2,-1,2)
    X_hom = cv2.triangulatePoints(Ps[0],Ps[1],ys[0].T,ys[1].T)
    X = (X_hom[:3] / X_hom[3]).T.reshape(*batch_shape,3)
    return jnp.array(X)

    
# =================================================================== 
# Coordinate frames
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

# =================================================================== 
# Angles and rotations
# =================================================================== 

def Rxy_mat(thetas):
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

def signed_angular_difference(a, b, n):
    """Compute signed angular differencee required to rotate a TO b,
    about the plane normal vector n.

    Input arrays expected to be unit length.

    Parameters
    ----------
        a: ndarray, shape (...,D)
            Source vector
        b: ndarray, shape (...,D)
            Destination vector
        n: ndarray, shape (D,) or (...,D)
            Normal vector about which to rotate

    Returns
    -------
        theta: ndarray, shape (...,)
    """
    cos_theta = jnp.dot(jnp.cross(a, b), n)
    sin_theta = jnp.einsum('...jk, ...jk -> ...j', a, b)
    return jnp.arctan2(cos_theta, sin_theta)

# =================================================================== 
# Gaussians with constrained precision matrices
# =================================================================== 

def children_of(parents):
    """Return list of children per node given list of parents

    Parameters
    ----------
        parents: 1D array of list, length (K,) of ints.
            parent[j] is index of parent node of node j.

    Returns
    -------
        children: list, length (K,).
            children[k] contains variable length list of children node
            indices. May be empty if node has no children.
    """

    K = len(parents)

    children = []
    for k in range(K):
        children.append([])
        for j in range(k+1, K):
            if parents[j] == k:
                children[k].append(j)

    return children

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

def hvmfg_natural_parameter(children, radii, variance, directions):
    """Calculate first natural parameter of hierarchical vMFG.

    TODO Incorporate into hvMFG distribution

    Parameters
    ----------
        children: list, length K
        radii: ndarray, shape (K,)
        variance: ndarray, shape (K,)
        directions: ndarray, shape (..., K, D)

    Returns
    -------
        h: ndarray, shape (..., K, D)

    """

    K = len(children)
    
    h = jnp.zeros_like(directions)
    
    for k in range(K):
        # Contributions from self
        h = h.at[...,k,:].add(directions[...,k,:] * (radii[k]/variance[k]))

        # Contributions from children
        for j in children[k]:
            h = h.at[...,k,:].add(-directions[...,j,:] * (radii[j]/variance[j]))

    return h

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
