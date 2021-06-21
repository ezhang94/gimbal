""""
gimbal/mcmc.py
"""

import jax.numpy as jnp
import jax.random as jr
from jax import jit, partial
from jax.scipy.special import logsumexp

import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd

from util import (triangulate, project,
                  xyz_to_uv, uv_to_xyz, signed_angular_difference,
                  Rxy_mat, cartesian_to_polar)

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
        if D_obs == 2:
            obs = jnp.moveaxis(observations, 1, 0)
            positions = triangulate(params['camera_matrices'], obs)
        elif D_obs == 3:
            positions = observations.copy()
        else:
            raise ValueError(f"Expected observation data dimension {{2,3}}, but receieved {D_obs}.")
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

def sample_positions(seed, params, observations, samples,
                     step_size=1e-1, num_leapfrog_steps=1):
    """Sample positions by taking one Hamiltonian Monte Carlo step.

    """
    
    N, C, K, D_obs = observations.shape

    raise NotImplementedError
    # TODO
    objective = partial(log_joint_probability,
                        samples['outliers'],
                        )

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=objective,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size
            )

    positions, kernel_results = hmc.one_step(
                last_positions, 
                hmc.bootstrap_results(last_positions),
                seed=seed)

    return positions, kernel_results

@jit
def sample_outliers(seed, params, observations, samples):
    """Sample outliers
    
    TODO define inlier/outlier distributions beforehand. These are static.
    """

    predicted_observations = jnp.stack(
        [project(P, samples['positions']) for P in params['camera_matrices']],
        axis=1)
    error = observations - predicted_observations

    # Log probability of predicted observation being inlier
    Y_inlier = tfd.MultivariateNormalFullCovariance(
                                    params['obs_inlier_location'], 
                                    params['obs_inlier_covariance']
                                    )
    lp_k0 = jnp.log(1-params['obs_outlier_probability'])
    lp_k0 += Y_inlier.log_prob(error)

    # Log probability of predicted observation being outlier
    Y_outlier = tfd.MultivariateNormalFullCovariance(
                                    params['obs_outlier_location'], 
                                    params['obs_outlier_covariance']
                                    )
    lp_k1 = jnp.log(params['obs_outlier_probability'])
    lp_k1 += Y_outlier.log_prob(error)
    
    # Update posterior
    lognorm = logsumexp(jnp.stack([lp_k0, lp_k1], axis=-1), axis=-1)
    p_isoutlier = jnp.exp(lp_k1 - lognorm)
    
    # Draw new samples
    outliers = jr.uniform(seed, observations.shape[:-1]) < p_isoutlier

    # Any NaN observations are obviously drawn from outlier distribution
    outliers = jnp.where(jnp.isnan(observations[...,0]), True, outliers)

    return outliers

@jit
def sample_directions(seed, params, samples):
    """Sample directions from 3D von Mises-Fisher distribution."""

    positions = samples['positions']    # (N,K,3)
    heading = samples['heading']        # (N,)
    state = samples['pose_state']       # (N,)
    D = positions.shape[-1]

    # Rotate mean directional prior (defined in the canonical reference 
    # frame) into the absolute/ambient reference frame
    mu_priors = jnp.einsum('tmn, tjn-> tjm',
                           Rxy_mat(heading), params['state_directions'][state])
    kappa_priors = params['state_concentrations'][state]

    # Keypoint position contributions to direction vectors
    dx = positions - positions[:, params['parents']]
    
    # Calculate posterior parameters
    mu_tilde  = kappa_priors[...,None] * mu_priors
    mu_tilde += dx * (params['pos_radius']/params['pos_radial_variance'])[:,None]
    
    mu_post = mu_tilde / jnp.linalg.norm(mu_tilde, axis=-1, keepdims=True)
    kappa_post = jnp.linalg.norm(mu_tilde, axis=-1)
    
    # Sample from posterior
    directions = tfd.VonMisesFisher(mu_post, kappa_post).sample(seed=seed)
    directions = directions.at[:,0,:].set(jnp.zeros(D))

    return directions

@jit
def sample_headings(seed, params, samples):
    """Sample headings from uniform circular distribution."""

    state = samples['pose_state']   # (N,)

    # Polar representation of 3D vectors, array shapes (N,K)
    mu_thetas, mu_phis = cartesian_to_polar(params['state_directions'][state][:,1:])
    us_thetas, us_phis = cartesian_to_polar(samples['directions'][:,1:])

    # Update parameters, which we find by solving
    #   k_likelihood sin(theta_likelihood) = sum w_i sin(theta_i)
    #   k_likelihood cos(theta_likelihood) = sum w_i cos(theta_i)
    # Recall: Our prior for headings is vMF with concentration 0
    # so, theta_post = theta_likelihood, k_post = k_likelihood
    azim_weights = jnp.sin(mu_phis) * jnp.sin(us_phis)
    k_sin_thetas = jnp.sum(azim_weights * jnp.sin(us_thetas - mu_thetas), axis=-1)
    k_cos_thetas = jnp.sum(azim_weights * jnp.cos(us_thetas - mu_thetas), axis=-1)

    theta_post = jnp.arctan2(k_sin_thetas, k_cos_thetas)
    kappa_post = jnp.sqrt(k_sin_thetas**2 + k_cos_thetas**2)
    
    return tfd.VonMises(theta_post, kappa_post).sample(seed=seed)
    