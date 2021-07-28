# Load other modules
import os, sys
import numpy as onp

import h5py
import yaml

import jax.random as jr

# Add GIMBAL source code to path
SRC = os.path.join(os.environ['HOME'], 'gimbal/gimbal')
sys.path.insert(0, SRC)

import run
from . import mcmc
from . import util_io

OUTPUTPATH = os.environ['SCRATCH']
PROJECTPATH = os.path.dirname(__file__)  # Project folder (where this script lives)
DATAPATH = PROJECTPATH  # Example data is stored in the project folder

skeleton_path = os.path.join(PROJECTPATH, 'skeleton.yml')
input_path = os.path.join(DATAPATH, 's1-d1-predictions.hdf5')
camera_path = input_path

# ========================================================================
# Load known camera calibration parameters
camera_matrices = util_io.load_camera_parameters(camera_path)

# Load specified skeleton
keypoint_names, parents = util_io.load_skeleton(skeleton_path)

# ========================================================================
# Training data
with h5py.File(input_path) as f:
    k_ordering = list(f.attrs['keypoint_names'])
    positions = f['training/mocap'][()]
    observations = f['training/dlc2d'][()]

col_to_tree_order = [k_ordering.index(kname) for kname in keypoint_names]
observations = observations[..., col_to_tree_order, :]
positions = positions[..., col_to_tree_order, :]

params = run.fit(positions, parents=parents,
                 observations=observations, camera_matrices=camera_matrices,
                 crf_keypoints=[10,0],
                 )

params['obs_outlier_probability'] \
                = params['obs_outlier_probability'][...,col_to_tree_order]
params['obs_outlier_variance'] \
                = params['obs_outlier_variance'][...,col_to_tree_order]
params['obs_inlier_variance'] \
                = params['obs_inlier_variance'][...,col_to_tree_order]

params['state_directions'] \
                = params['state_directions'][...,col_to_tree_order,:]
params['state_concentrations'] \
                = params['state_concentrations'][...,col_to_tree_order]

# ========================================================================
# Predict
seed = jr.PRNGKey(1916)
num_mcmc_iterations = 2000
start_idx, end_idx = 2700, 6000
outpath = os.path.join(OUTPUTPATH, f'f{start_idx}_{end_idx}')

# Load observations
with h5py.File(input_path) as f:
    k_ordering = list(f.attrs['keypoint_names'])
    observations_v = f['validation/dlc2d'][start_idx:end_idx]

# Modify imported data to match skeleton definition
col_to_tree_order = [k_ordering.index(kname) for kname in keypoint_names]
observations_v = observations_v[..., col_to_tree_order, :]

out = mcmc.predict(seed, params, observations_v,
                   num_mcmc_iterations=num_mcmc_iterations,
                   enable_x64=True,
                   out_options={'path':outpath, 'chunk_size':500, 'burnin': 1000,
)