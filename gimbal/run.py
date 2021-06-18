import os
import numpy as onp
import jax.numpy as jnp
import jax.random as jr

import mcmc
from util import (tree_graph_laplacian,)

# =====================================================================

def _fit_obs_error_parameters(positions, observations, camera_matrices):
    # TODO

    print('Fitting observation error parameters...')

    fpath = os.path.join(os.environ['DATAPATH'],
                         'gmm_obs_error-dlc2d_mu0_filtered.npz')
    with onp.load(fpath, 'r') as f:
        m_fitted = jnp.asarray(f['means'])
        k_fitted = jnp.asarray(f['sigmasqs'])
        w_fitted = jnp.asarray(f['p_isOutlier'])

    return dict(obs_outlier_probability=w_fitted,
                obs_outlier_location=0.,
                obs_outlier_variance=k_fitted[...,0],
                obs_inlier_location=0.,
                obs_inlier_variance=k_fitted[...,1],
                camera_matrices=camera_matrices)

def _fit_skeletal_parameters(positions, parents, root_variance=1e8):
    """Estimate inter-keypoint distances and variances.

    Note that inter-keypoint distance and variance is technically
    undefined for the root node (k=0), but we retain a placeholder value
    for subsequent consistency in indexing.

    Returns
    -------
        dict, with values
            pos_radius: ndarray, shape (K, )
            pos_radial_variance: ndarray, shape (K, )
            parents: length K
    """

    print('Fitting skeletal parameters...')
    dx = jnp.linalg.norm(positions - positions[:, parents, :], axis=-1)
    dx_mean = jnp.nanmean(dx, axis=0)
    
    dx_variance = jnp.nanvar(dx, axis=0)
    dx_variance = dx_variance.at[0].set(root_variance)

    return dict(pos_radius=dx_mean,
                pos_radial_variance=dx_variance,
                parents=parents,)

def _fit_time_smoothing_parameters(positions):
    """Estimate positional variance of each keypoint from frame to frame.

    Calculate variance as the squared average distance between positions
    in subsequent frames.

    Returns
    -------
        dict, with item
            pos_dt_variance: ndarray, shape (K, )
    """

    print('Fitting time smoothing parameters...')

    dx = jnp.linalg.norm(positions[1:] - positions[:-1], axis=-1)

    return dict(pos_dt_variance=jnp.nanmean(dx, axis=0)**2)

def _fit_pose_state_parameters(positions, parents, crf_keypoints,
                               crf_abscissa=jnp.array([1.,0, 0]),
                               crf_normal=jnp.array([0, 0, 1.]),
                               ):
    # TODO
    print('Fitting pose state parameters...')

    fpath = os.path.join(os.environ['DATAPATH'],
                         'directional_priors_filtered_s180_maxk200.npz')
    
    with onp.load(fpath, 'r') as f:
        dir_priors_pis = jnp.asarray(f['pis'])
        dir_priors_mus = jnp.asarray(f['mus'])
        dir_priors_kappas = jnp.asarray(f['kappas'])
        num_states = jnp.asarray(f['num_states'])

    return dict(state_probability=dir_priors_pis,
                state_directions=dir_priors_mus,
                state_concentrations=dir_priors_kappas,
                parents=parents,
                crf_keypoints=crf_keypoints,
                crf_abscissa=crf_abscissa,
                crf_normal=crf_normal)

# =====================================================================

def fit(positions,
        parents=None, root_variance=1e8,
        observations=None, camera_matrices=None,
        crf_keypoints=None,
        crf_abscissa=jnp.array([1.,0, 0]), crf_normal=jnp.array([0, 0, 1.]),
        parameters_to_fit=[], outpath=None,
        ):
    """
    Parameters
    ----------
        positions: ndarray, shape (N, K, 3)
            Ground truth keypoint positions. May contain NaNs.
        parents: list of ints, length K, optional.
            parents[k] is the parent index of the k-th keypoint.
            Required to fit skeletal parameters
        observations: ndarray, shape (N, C, K, D_obs), optional.
            Input keypoint predictions, to be refined, D_obs = {2,3}
            Required to fit observation error GMM parameters
        camera_matrices: ndarray, shape (C, D_obs+1, 4), optional.
            Camera projection matrices.
        crf_keypoints: tuple, length 2
            Keypoints specify base and tip of vector to align to 
            the canonical direction (i.e. `crf_abscissa`)
        crf_abscissa: ndarray, shape (3,)
            (Unit) vector identifying the first axis/abscissa of the
            canonical reference frame. default: x-axis
        crf_normal: ndarray, shape (3,)
            Unit vector identifying the normal/applicate direction of the
            canonical reference frame. default: z-axis
        parameters: list, str
            Specify subset of parameters to load. If empty (default):
            fit all parameters
        outpath: str
            Path where parameters should be saved.
            default: None, do not save
        
    Returns
    -------
        params: dict, with keys
            camera_matrices, parents,
            obs_outlier_probability,
            obs_outlier_mean, obs_outlier_variance,
            obs_inlier_mean, obs_inlier_variance,
            pos_radius, pos_radial_variance,
            pos_dt_variance, pos_dt_variance_0, pos_dynamic_mean_0,
            crf_keypoints, crf_abscissa, crf_normal,
            state_probability, state_directions, state_concentrations,
            state_transition_count,
    """

    # If no parameters to fit specified, fit all
    if not parameters_to_fit:
        parameters_to_fit = [
            'observation_error',
            'skeletal_distance',
            'temporal_smoothness',
            'pose_state',]

    warn_missing_input= lambda pkey, input: \
        print(f"WARNING: Missing input '{input}'. Cannot fit {pkey} parameters...Skipping.")
    
    params = {}
    for pkey in parameters_to_fit:
        if pkey == 'observation_error':
            if observations is None: warn_missing_input(pkey, 'observations')
            elif camera_matrices is None: warn_missing_input(pkey, 'camera_matrices')
            else:
                params.update(
                    _fit_obs_error_parameters(positions,
                                              observations,
                                              camera_matrices)
                )
        elif pkey == 'skeletal_distance':
            if parents is None: warn_missing_input(pkey, 'parents')
            else:
                params.update(
                    _fit_skeletal_parameters(positions,
                                             parents,
                                             root_variance)
                )
        elif pkey == 'temporal_smoothness':
            params.update(
                _fit_time_smoothing_parameters(positions)
            )
        elif pkey == 'pose_state':
            if parents is None: warn_missing_input(pkey, 'parents')
            elif crf_keypoints is None:warn_missing_input(pkey, 'crf_keypoints')
            else:
                params.update(
                    _fit_pose_state_parameters(positions,
                                               parents,
                                               crf_keypoints,
                                               crf_abscissa,
                                               crf_normal)
                )
        else:
            print(f"WARNING: Unexpected parameter specification '{pkey}'. Skipping.")

    return params

# =====================================================================

def standardize_parameters(params,
                           pos_location_0=0., pos_variance_0=1e8,
                           state_transition_count=10.):
    num_keypoints = len(params['parents'])
    num_cameras =  len(params['camera_matrices'])
    dim = params['camera_matrices'].shape[-1] - 1
    dim_obs = params['camera_matrices'].shape[-2] - 1
    num_states = len(params['state_probability'])

    # -----------------------------------
    # Canonical reference frame
    # -----------------------------------
    x_axis, z_axis = params['crf_abscissa'], params['crf_normal']
    y_axis = jnp.cross(z_axis, x_axis)
    params['crf_axes'] = jnp.stack([x_axis, y_axis, z_axis], axis=0)

    # -----------------------------------
    # Skeletal and positional parameters
    # -----------------------------------
    params['pos_radius'] \
        = jnp.broadcast_to(params['pos_radius'], (num_keypoints,))

    params['pos_radial_variance'] \
        = jnp.broadcast_to(params['pos_radial_variance'], (num_keypoints,))
    params['pos_radial_precision'] \
        = tree_graph_laplacian(
            params['parents'],
            1 / params['pos_radial_variance'][...,None,None] * jnp.eye(dim)
            )
    params['pos_radial_covariance'] \
        = jnp.linalg.inv(params['pos_radial_precision'])

    params['pos_dt_variance'] \
        = jnp.broadcast_to(params['pos_dt_variance'], (num_keypoints,))
    params['pos_dt_covariance'] \
        = jnp.kron(jnp.diag(params['pos_dt_variance']), jnp.eye(dim))
    params['pos_dt_precision'] \
        = jnp.kron(jnp.diag(1./params['pos_dt_variance']), jnp.eye(dim))

    params['pos_precision_t'] \
        = params['pos_radial_precision'] + params['pos_dt_precision']
    params['pos_covariance_t'] \
        = jnp.linalg.inv(params['pos_precision_t'])

    params['pos_variance_0'] \
        = jnp.broadcast_to(
                params.get('pos_variance_0', pos_variance_0),
                (num_keypoints,))
    params['pos_covariance_0'] \
        = jnp.kron(jnp.diag(params['pos_variance_0']), jnp.eye(dim))

    params['pos_location_0'] \
        = jnp.broadcast_to(
                params.get('pos_location_0', pos_location_0),
                (num_keypoints, dim))

    # -----------------------------
    # Observation error parameters
    # -----------------------------
    params['obs_outlier_variance'] \
        = jnp.broadcast_to(
                params['obs_outlier_variance'],
                (num_cameras, num_keypoints))
    params['obs_outlier_covariance'] \
        = jnp.kron(
                params['obs_outlier_variance'][..., None, None],
                jnp.eye(dim_obs))

    params['obs_inlier_variance'] \
        = jnp.broadcast_to(
                params['obs_inlier_variance'],
                (num_cameras, num_keypoints))
    params['obs_inlier_covariance'] \
        = jnp.kron(
                params['obs_inlier_variance'][..., None, None],
                jnp.eye(dim_obs))

    params['obs_inlier_location'] \
        = jnp.broadcast_to(
                params['obs_inlier_location'],
                (num_cameras, num_keypoints))
    params['obs_outlier_location'] \
        = jnp.broadcast_to(
                params['obs_outlier_location'],
                (num_cameras, num_keypoints))
    
    params['obs_outlier_probability'] \
        = jnp.broadcast_to(
                params['obs_outlier_probability'],
                (num_cameras, num_keypoints))

    # -----------------------------
    # Pose state parameters
    # -----------------------------
    params['state_transition_count'] \
        = jnp.broadcast_to(
                params.get('state_transition_count', state_transition_count),
                (num_states,))
    
    params['state_probability'] \
        = jnp.broadcast_to(
                params.get('state_probability', 1./num_states),
                (num_states,))

    params['state_directions'] \
        = jnp.broadcast_to(
                params.get('state_directions', jnp.array([0,0,1.])),
                (num_states, num_keypoints, dim))

    params['state_concentrations'] \
        = jnp.broadcast_to(
                params.get('state_concentrations', 0.),
                (num_states, num_keypoints))
    
    return params

# =====================================================================

def predict(seed, params, observations, init_positions=None,
            num_mcmc_iterations=1000,
            hmc_options={'init_step_size':1e-1, 'num_leapfrog_steps':1},
            out_options={}
            ):
    """Predict latent variables from observations using MCMC sampling.

    Parameters
    ----------
        seed: jax.random.PRNGKey
        params: dict
        observations: ndarray, shape (N, C, K, D_obs)
        init_positions: None or ndarray, shape (N, K, D), optional
            Initial guess of 3D positions. If None (default), initial
            guess made from observations in mcmc.initialize
        num_mcmc_iterations: int
        hmc_options: dict, optional
        out_options: dict, optional
            If empty (default), samples are not saved. Else, samples are
            saved to HDF5 according to the specified items:
                path: str
                chunksize:
                thinning: int, optional.
                burnin: int, optional
                variables: list, optional
            default: {}, do not save.

    Returns
    -------
    """

    params = standardize_parameters(params)

    seed = iter(jr.split(seed))
    samples = mcmc.initialize(next(seed), params,
                              observations, init_positions)

    return samples


if __name__ == "__main__":
    pass