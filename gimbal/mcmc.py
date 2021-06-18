""""
gimbal/mcmc.py
"""

import jax.numpy as jnp
import jax.random as jr

from util import (triangulate, xyz_to_uv, uv_to_xyz,
                  signed_angular_difference)

def initialize(seed, params, observations, init_positions=None):
    """Initialize latent variables of model.

    Parameters
    ----------
        seed: jax.random.PRNGKey
        params: dict
        observations: ndarray, shape (N, C, K, D_obs)
        init_positions: ndarray, shape (N, K, D), optional.
            Initial guess of 3D positions. If None (default), positions
            are triangulated using direct linear triangulation.
        
    Returns
    -------
        samples: dict
    """

    seed = iter(jr.split(seed, 3))

    N, C, K, D_obs = observations.shape

    # ---------------------
    # Initialize positions
    # ---------------------
    if init_positions is None:
        positions = triangulate_multiview(params['camera_matrices'],
                                          observations)
    else:
        positions = jnp.asarray(init_positions)

    # --------------------------
    # Derive initial directions
    # --------------------------
    directions = positions - positions[:,params['parents']]
    directions /= jnp.linalg.norm(directions, axis=-1, keepdims=True)

    # Use placeholder value for undefined root direction
    directions = directions.at[:,0].set(0.)

    # -------------------------------
    # Derive initial heading vectors
    # -------------------------------
    k_base, k_tip = params['crf_keypoints']
    u_axis, v_axis, n_axis = params['crf_axes']
    
    # Unnormalized direction vector
    heading = positions[:,k_tip] - positions[:,k_base]

    # Project 3D direction vectors onto 2D plane of coordinate reference
    # frame (CRF). Then, re-embed the 2D plane in the ambient 3D frame.
    # In the default case, the CRF is defined as the standard Cartesian
    # reference (i.e. poses aligned to "x"-axis, and headings considered
    # in the "x-y" plane, rotating about normal "z"-axis.) The following
    # two operations then effectively zero-out the z-axis coordinates.
    heading = xyz_to_uv(heading, u_axis, v_axis)
    heading = uv_to_xyz(heading, u_axis, v_axis)

    # Normalize vector to unit length
    heading /= jnp.linalg.norm(heading, axis=-1, keepdims=True)

    # Calculate angle from canonical reference direction
    heading = signed_angular_difference(
                jnp.broadcast_to(u_axis, heading.shape), heading, n_axis)

    # ---------------------------
    # Sample outliers from prior
    # ---------------------------
    outliers = jr.uniform(next(seed), (N,C,K)) < params['obs_outlier_probability']

    # Consider any NaN observations as outliers
    outliers = jnp.where(jnp.isnan(observations).any(axis=-1),
                         True, outliers)

    # ------------------------------------------------------
    # Sample pose states and transition matrix from uniform
    # ------------------------------------------------------
    # TODO sample from prior parameters, state_probability and state_counts
    num_states = len(params['state_probability'])
    pose_state = jr.randint(next(seed), (N,), 0, num_states)
    transition_matrix = jnp.ones((num_states, num_states)) / num_states

    return dict(
        outliers=outliers,
        positions=positions,
        directions=directions,
        heading=heading,
        pose_state=pose_state,
        transition_matrix=transition_matrix,
    )

