**Dataset name**

`s1-d1-predictions.hdf5`

**Dataset description**

Training and validation keypoint datasets for Rat7M s1-d1.
Contains ground truth filter motion capture data, and
predictions by DLC2D, DLC3D, and DANNCE networks.

Also includes calibrated camera parameters.

**Source**

```
Marshall, Jesse D.; Aldarondo, Diego; Wang, William; P. Ölveczky, Bence; Dunn, Timothy (2021): Rat 7M.
figshare. Collection. https://doi.org/10.6084/m9.figshare.c.5295370.v3 
```

**File structure**
```
s1-d1-predictions
+-- attrs
|   +-- session
|   +-- keypoint_names
|   +-- camera_names
+-- camera
|   +-- intrinsisc              # shape (6,3,3), <f8>
|   +-- rotation                # shape (6,3,3), <f8>
|   +-- translation             # shape (6,3), <f8>
|   +-- radial_distortion       # k1 and k2 distortion parameters, <f8>
|   +-- tangential_distortion   # p1 and p2 distoriton parameters, <f8>
|-- training
|   +-- mocap                   # shape (54000,20,3), <f8>
|   +-- dannce                  # shape (54000,20,3), <f8>
|   +-- dlc2d                   # shape (54000,6,20,3), <f8>
|   +-- dlc3d                   # shape (54000,20,3), <f8>
|   +-- sampleID                # shape (54000,), <u8>
|-- validation
|   +-- mocap                   # shape (12800,20,3), <f8>
|   +-- dannce                  # shape (12800,20,3), <f8>
|   +-- dlc2d                   # shape (12800,6,20,3), <f8>
|   +-- dlc3d                   # shape (12800,20,3), <f8>
|   +-- sampleID                # shape (12800,), <u8>
```

**Other notes**

| Partition    | `sampleID` start | `sampleID` end |
| ------------ | ---------------- | -------------- |
| `training`   | 1973266          | 3773231        |
| `validation` | 3773264          | 4199897        |