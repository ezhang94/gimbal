"""
gimbal/util_io.py

Utility functions for file input and out

"""

import numpy as onp
import jax.numpy as jnp

import h5py
import yaml

def load_skeleton(fpath):
    """Load keypoint and parent specification defining skeletal stucture.

    Parameters
    ----------
        fpath: file-like object, string, or pathlib.Path
            .yml file to read from

    Returns
    -------
        keypoint_names: str list, len K
        parents: int list, len K

    """
    with open(fpath) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        skeleton = config['skeleton']

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
        fpath: str or tuple , path to HDF5 file with camera parameters
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

    if isinstance(fpath, str):
        fpath = fpath
        key = ''
        assert isinstance(fpath, str)
    elif isinstance(fpath, tuple):
        if len(fpath) == 1:
            fpath = fpath[0]
            assert isinstance(fpath, str)
        elif len(fpath) == 2:
            fpath, key = fpath
            assert isinstance(fpath, str)
            assert isinstance(key, str)
        else:
            raise ValueError(f'Expect fpath tuple to consist of (file, key) pair, but got tuple of length {len(fpath)}.')    
    else:
        raise ValueError(f'Expected fpath to be str or (str, str) tuple, but got {type(fpath)}.')


    # If no cameras specified, load all cameras
    c_idxs = onp.s_[:] if not cameras else cameras

    with h5py.File(fpath) as f:
        intrinsic = f[key]['intrinsic'][c_idxs]
        rotation = f[key]['rotation'][c_idxs]
        translation = f[key]['translation'][c_idxs]

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

def save_parameters(file, params):
    """Save GIMBAL parameters dictionary to compressed .npz file.

    Wrapper function around numpy.savez_compressed. The reasons are:
        - jax.numpy.savez is just a wrapper function around numpy.savez
        - Both jax and numpy implementations of savez* convert 
          jaxlib.xla_extension.DeviceArray type to numpy.ndarray type
        - numpy.savez_compressed offers ~3x compression over numpy.savez
        - jax.numpy.savez_compressed is not yet implemented

    Parameters
    ----------
        file: file-like object, string, or pathlib.Path
            File to write to.
        params: dict

    Returns
    -------
        None
    """
    onp.savez_compressed(file, **params)
    return

def load_parameters(file, to_device_array=True):
    """Load saved GIMBAL parameters from .npz file

    Note that both jax.numpy.load loads .npz file of numpy.ndarrays,
    so jax.x

    Parameters
    ----------
        file: file-like object, string, or pathlib.Path
            .npz file to read from.
        to_device_array: bool, optional.
            If True (default), casts arrays as jaxlib.xla_extension.DeviceArray.
            Else, cast all arrays as numpy.ndarray.

    Returns
    -------
        params: dict
    """
    
    np = jnp if to_device_array else onp
    
    params = {}
    with np.load(file) as f:
        for k in list(f):
            params[k] = np.asarray(f[k])

    return params

# -------------------
# Saving predictions
# -------------------

class SavePredictionsToHDF():
    def __init__(self, path, init_samples,
                 max_iter=None,chunk_size=None,
                 mode='w', hdf_kwargs={}):
        
        self.path = path
        self._obj = h5py.File(path, mode, **hdf_kwargs)
        
        for k,v in init_samples.items():
            if isinstance(chunk_size, int):
                chunks = (chunk_size, *v.shape)
            else:
                chunks = (max_iter, *v.shape)
            self._obj.create_dataset(k, (0, *v.shape), dtype=v.dtype,
                                  maxshape=(max_iter, *v.shape),
                                  chunks=chunks,
                                  compression='gzip')
    
    @property
    def obj(self,):
        return self.path
    
    def __enter__(self,):
        return self
    
    def __exit__(self,exc_type, exc_value, tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, tb)
        self._obj.close()
        return
    
    def close(self):
        self._obj.close()

    def update(self, buffer):
        """Update HDF datasets with new samples."""
    
        for k, v in buffer.items():
            N = len(v)
            self._obj[k].resize(len(self._obj[k]) + N, axis=0)
            self._obj[k][-N:] = jnp.stack(v, axis=0)
        return