# CPU Installation

```
    python3 -m pip install jax jaxlib
    python3 -m pip install tfp-nightly[jax]
```

# GPU Installation

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