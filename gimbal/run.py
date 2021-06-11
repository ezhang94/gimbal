import os
import numpy as onp

# =====================================================================
# TEMPORARY
# If DATAPATH environmental variable not set, default to CWD
try:
    DATAPATH = os.environ['DATAPATH']
except:
    print("WARNING: DATAPATH variable not found. Defaulting to current working directory.")
    DATAPATH = os.getcwd()
print("DATAPATH: ", DATAPATH)

config_file_path = os.path.join(os.path.dirname(__file__), '../scripts/s1-d1/config.yml')
cparams_path = os.path.join(DATAPATH, "s1-d1-predictions.hdf5")
mocap_path = os.path.join(DATAPATH, "s1-d1-predictions.hdf5")
gmm_path = os.path.join(DATAPATH, "gmm_obs_error-dlc2d_mu0_filtered.npz")
em_path = os.path.join(DATAPATH, "directional_priors_filtered_s180_maxk200.npz")
# =====================================================================

def load_parameters(mocap_path, gmm_path, em_path):
    global DATAPATH

    # GMM inlier/outlier parameters
    # Hires2/JDM31 with filtered mocap data, separated heads
    with onp.load(gmm_path, 'r') as f:
        m_fitted = f['means']
        k_fitted = f['sigmasqs']
        w_fitted = f['p_isOutlier']

    # Pose state parameters
    with onp.load(em_path, 'r') as f:
        print(f.files)
        dir_priors_pis = f['pis']
        dir_priors_mus = f['mus']
        dir_priors_kappas = f['kappas']
        num_states = f['num_states']


if __name__ == "__main__":    
    load_parameters()
    pass