import os
import numpy as onp
import jax.numpy as jnp
import jax.random as jr

import mcmc
import util_io

from tqdm.auto import trange

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
                obs_outlier_variance=k_fitted[...,1],
                obs_inlier_location=0.,
                obs_inlier_variance=k_fitted[...,0],
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

def predict(seed, params, observations, init_positions=None,
            num_mcmc_iterations=1000,
            hmc_options={'init_step_size':1e-1, 'num_leapfrog_steps':1},
            out_options={},
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
                chunk_size:
                thinning: int, optional.
                burnin: int, optional
                variables: list, optional
            default: {}, do not save.

    Returns
    -------
    """

    # Initialize
    seed, init_seed = jr.split(seed, 2)

    params = mcmc.initialize_parameters(params)
    samples = mcmc.initialize(init_seed, params,
                              observations, init_positions)
    samples['log_probability'] = \
            mcmc.log_joint_probability(params, observations, *samples.values())

    # Enable timing, if specified. Add appropriate key to `init_dict` to save.

    # Enable chunked saving, if specified
    init_dict = samples
    if not out_options: # If no out options specified
        Fout = util_io.SavePredictionsToDict(init_dict)
        chunk_size = num_mcmc_iterations
    else:
        Fout = util_io.SavePredictionsToHDF(
                    out_options['path'], init_dict,
                    max_iter = num_mcmc_iterations,
                    **out_options.get('hdf_kwargs', {}))
        chunk_size = out_options.get('chunk_size', num_mcmc_iterations)

    # Setup progress bar
    pbar = trange(num_mcmc_iterations)
    pbar.set_description("lp={:.4f}".format(samples['log_probability']))

    buffer = {k: jnp.empty((chunk_size, *v.shape), dtype=v.dtype)
              for k, v in init_dict.items()}
    with Fout as out:
        for itr in pbar:

            samples, kernel_results = \
                mcmc.step(jr.fold_in(seed, itr), params, observations, samples)

            # ------------------------------------------------------------------
            # Update the progress bar
            pbar.set_description("lp={:.4f}".format(samples['log_probability']))
            pbar.update(1)

            # Save
            for k, v in buffer.items():
                buffer[k] = buffer[k].at[itr % chunk_size].set(samples[k])
            
            if (chunk_size is not None) and ((itr+1) % chunk_size == 0):
                out.update(buffer)
        
        
        # Store final chunk of samples, if needed
        if (chunk_size is None) or (itr+1) % chunk_size != 0:
            for k, v in buffer.items():
                buffer[k] = buffer[k][:itr % chunk_size + 1]
            
            out.update(buffer)
    
    # import pdb
    # pdb.set_trace()

    return samples, Fout.obj

if __name__ == "__main__":
    pass