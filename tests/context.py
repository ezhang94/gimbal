import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../gimbal/')))

import distributions
import util

# test_data contains the following corresponding data:
#     keypoint_names: array of str, length K
#     parents: array of int, length K
#     camera_matrices: ndarray, shape (C,3,4)
#         Calibrated camera matrices
#     observed_pos_2d: ndarray, shape (N,C,K,2)
#         2D observations of positions, predicted by DeepLabCut network
#     triangulated_pos_3d: ndarray, shape (N,K,3)
#         3D positions triangulated from observed_pos_2d using pairwise DLT method and then taking median across pairs
#     groundtruth_pos_3d: ndarray, shape(N,K,3)
#         Ground truth 3D positions, measured with motion capture system
# where N=100, C=6, K=18.
TESTS_DIR = os.path.dirname(__file__)
DATA = os.path.join(TESTS_DIR, 'mock_data.npz')
PARAMS = os.path.join(TESTS_DIR, 'mock_params.npz')
