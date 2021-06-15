import yaml
import h5py

import numpy as onp
import jax.numpy as jnp

def load_skeleton(skeleton):
    """Load keypoint and parent specification defining skeletal stucture.

    Parameters
    ----------
        skeleton: dict, ordered {keypoint:parent} dictionary entries

    Returns
    -------
        keypoint_names: str list, len K
        parents: int list, len K

    """
    keypoint_names = list(skeleton.keys())
    parent_names = list(skeleton.values())

    _k_type = type(keypoint_names[0])
    assert _k_type == type(parent_names[0]), \
        f"Skeleton key/values must be same type. Received '{_k_type}', '{type(parent_names[0])}'."
    assert isinstance(keypoint_names[0], (int, str)), \
        f"Skeleton key/values must be int or string. Received '{_k_type}'."

    if _k_type is str:
        # Identify index of corresponding parent keypoint name
        parents = [keypoint_names.index(name) for name in parent_names]
    else:
        # Create placeholder names for keypoints
        keypoint_names = [f'k{i}' for i in range(len(keypoint_names))]
        parents = parent_names

    assert parents[0] == 0, \
        f"Parent of root node should be itself. Received '{parents[0]}'."
    return keypoint_names, parents

def load_camera_parameters(fpath, cameras=[], mode='array'):
    """Load camera parameters.

    Parameters
    ----------
        fpath: str, path to HDF5 file with camera parameters
        cameras: int list of cameras to load. optional
            If none specified, load all cameras.
            TODO: Allow selection by camera name (i.e. str list)
        mode: str, one of {'dict', 'array'}
            Specifies the return type of cparams (see below)

    Returns
    -------
        cparams: dict of intrinsic and extrinsic camera matrices
            If mode == 'dict', return dictionary of parameters with
            keys 'instrinsic', 'rotation', and 'translation'
            If mode == 'array', return matrix of shape (C, 3, 4)
    """

    # If no cameras specified, load all cameras
    c_idxs = onp.s_[:] if not cameras else cameras

    with h5py.File(fpath) as f:
        intrinsic = f['camera_parameters']['intrinsic'][c_idxs]
        rotation = f['camera_parameters']['rotation'][c_idxs]
        translation = f['camera_parameters']['translation'][c_idxs]

    if mode == 'array':
        # Camera projection matrix = [KR | Kt], shape (num_cameras, 3, 4)
        KR = onp.einsum('...ij, ...jk -> ...ik', intrinsic, rotation)
        Kt = onp.einsum('...ij, ...j -> ...i', intrinsic, translation)
        cparams = onp.concatenate([KR, Kt[:,:,None]], axis=-1)
    else:
        cparams = {}
        cparams['intrinsic'] = intrinsic
        cparams['rotation'] = rotation
        cparams['translation'] = translation

    return cparams