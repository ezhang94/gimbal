import yaml

def load_skeleton(config_file_path):
    """Load keypoint and parent specification defining skeletal stucture.

    Reads from a YAML file with skeleton defined as ordered
    {keypoint: parent} dictionary entries.

    Outputs
    -------
        keypoint_names: list, len K; str
        parents: list, len K; int

    """
    with open(config_file_path) as f:
        out = yaml.load(f, Loader=yaml.SafeLoader)
        skeleton = out['skeleton']

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