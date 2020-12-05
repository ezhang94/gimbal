# GIMBAL: GeometrIc Manifolds for Body Articulation and Localization

GIMBAL is an animal pose estimation model that uses recent advances in spherical manifold learning to capture the spatiotemporal constraints of body posture
to **accurately infer latent 3D keypoint positions** from 2D or 3D keypoint observations.
This model is applicable to animals with rigid, articulated skeletons.

GIMBAL employs a hierarchical von Mises-Fisher-Gaussian model (h-vMFG) to capture distance, directional, and temporal constraints.
It admits a simple Monte Carlo Markov Chain (MCMC) algorithm for approximate Bayesian inference that
produces posterior estimates and variances of 3D positions, joint directions, heading (relative to user-defined direction), and pose state.
