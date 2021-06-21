# jax
## CPU Installation

```
    python3 -m pip --upgrade pip
    python3 -m pip install jax jaxlib
    python3 -m pip install tfp-nightly[jax]
```

## GPU Installation

GPU availability can be checked from script by executing the follwing:
```
    if [ "$(command -v nvidia-smi)" ]; then
        # nvidia-smi command found, gpu exists
        ...
    else
        # nvidia-smi command NOT found, gpu does not exist
        ....
    fi
```

**TBC**

# ssm
Instructions found at the [SSM repo page](https://github.com/lindermanlab/ssm).

Navigate to a local executable direction, e.g. `~/.local/bin`. Then, as instructed,
```
    git clone git@github.com:slinderman/ssm.git
    cd ssm
    pip install numpy cython
    pip install -e .
```