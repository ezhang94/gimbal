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

# -------------------
# Saving predictions
# -------------------

class SavePredictions():
    """Base class for saving GIMBAL predictions"""
    def __init__(self):
        self._obj = None

    @property
    def obj(self,):
        return self._obj

    def __enter__(self,):
        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        return

    def update(self, samples):
        """Update storage object with latest samples.

        Parameters
        ----------
            samples: dict of lists of ndarrays
        Returns
        -------
            None
        """
        raise NotImplementederror

class SavePredictionsToHDF(SavePredictions):
    def __init__(self, path, init_samples, max_iter=None,
                 mode='w', hdf_kwargs={}):
        
        self.path = path
        self._obj = h5py.File(path, mode, **hdf_kwargs)
        
        for k,v in init_samples.items():
            self._obj.create_dataset(k, (0, *v.shape), dtype=v.dtype,
                                  maxshape=(max_iter, *v.shape),
                                  compression='gzip')
    
    @property
    def obj(self,):
        return self.path
    
    def __exit__(self,exc_type, exc_value, tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, tb)
        self._obj.close()
        return
    
    def update(self, samples):
        """Update HDF datasets with new samples."""
    
        for k, v in samples.items():
            N = len(v)
            self._obj[k].resize(len(self._obj[k]) + N, axis=0)
            self._obj[k][-N:] = jnp.stack(v, axis=0)
        return

class SavePredictionsToDict(SavePredictions):
    def __init__(self, init_samples):
        self._obj = dict.fromkeys(init_samples.keys(), [])
    
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, tb)
        # Upon exit, stack all samples along first axis into single ndarray.
        try:
            for k, v in self._obj.items():
                self._obj[k] = jnp.stack(v, axis=0)
        except:
            pass
        return

    def update(self, samples):
        """Update dictionary with new samples."""
        for k, v in samples:
            self._obj[k].extend(v)
        return