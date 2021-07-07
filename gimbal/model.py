import os
import numpy as onp
import jax.numpy as jnp
import jax.random as jr

from sklearn.mixture import GaussianMixture

import util
import util_io

# =====================================================================

def _fit_obs_error_parameters(positions, observations, camera_matrices,
                              num_mixtures=2, 
                             ):
    """Estimate parameters of Gaussian model of observation error.

    Initialization of parameters plays a significant role in the GaussianMixture
    fitting procedure. In particular, assume that there is no bias in the mean,
    and the covariance parameters between mixture distributions is distinct.

    Parameters
    ----------
        positions: ndarray, shape (N, K, D)
            Ground truth keypoint positions. May contain NaNs.
        observations: ndarray, shape (N, C, K, D_obs)
            Noisy and potentially corrupted observations
        camera_matrices: ndarray, shape (C, D_obs+1, D+1)
        num_mixtures: int
            Number of mixtures to fit. Default: 2
    """

    print('Fitting observation error parameters...')

    _, C, K, D_obs = observations.shape
    M = num_mixtures

    positions_projected = onp.stack([
        util.project(P, positions) for P in camera_matrices], axis=0)

    # Initial guesses

    # Allocate arrays
    # Since there are only 2 mixtures, we only need to store P(is_outlier)
    fitted_weights   = onp.empty[(C, K,)] 
    fitted_means     = onp.empty([C, K, D_obs, M])
    fitted_variances = onp.empty([C, K, D_obs])

    for c in range (C):
        for k in range(K):
            obs_err = observations_error[:,c,k,:]
            mask = ~onp.isnan(obs_err[...,0])

            gmm = GaussianMixture(
                        n_components=M,
                        covariance_type='spherical',
                        weights_init=[1-p_outlier[c,k], p_outlier[c,k]],
                        means_init=onp.zeros((M,D_obs))
                        precision_init=[1/omega_in**2, 1/omega_out**2],
                        ).fit(obs_err[mask])

            fitted_weights[c,k]   = gmm.weights_[-1]
            fitted_means[c,k]     = gmm.means_.T
            fitted_variances[c,k] = gmm.covariances_
    
    return dict(obs_outlier_probability=fitted_weights,
                obs_outlier_location=0.,
                obs_outlier_variance=fitted_variances[...,1],
                obs_inlier_location=0.,
                obs_inlier_variance=fitted_variances[...,0],
                camera_matrices=camera_matrices)

    # fpath = os.path.join(os.environ['HOME'],
    #                      'data/gmm_obs_error-dlc2d_mu0_filtered.npz')
    # with onp.load(fpath, 'r') as f:
    #     m_fitted = jnp.asarray(f['means'])
    #     k_fitted = jnp.asarray(f['sigmasqs'])
    #     w_fitted = jnp.asarray(f['p_isOutlier'])

    # return dict(obs_outlier_probability=w_fitted,
    #             obs_outlier_location=0.,
    #             obs_outlier_variance=k_fitted[...,1],
    #             obs_inlier_location=0.,
    #             obs_inlier_variance=k_fitted[...,0],
    #             camera_matrices=camera_matrices)

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

    fpath = os.path.join(os.environ['HOME'],
                         'data/directional_priors_filtered_s180_maxk200.npz')
    
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