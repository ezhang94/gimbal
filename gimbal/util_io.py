"""
gimbal/util_io.py

Utility functions for file input and out

"""

import numpy as onp
import jax.numpy as jnp

import h5py

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