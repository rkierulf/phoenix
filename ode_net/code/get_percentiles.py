import numpy as np
import os

#From https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-024-03264-0/MediaObjects/13059_2024_3264_MOESM2_ESM.pdf

def get_percentiles(output_root_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dynamics_mat_path = os.path.join(current_dir, output_root_dir, 'dynamics_mat.csv')
    dynamics_mat = np.loadtxt(dynamics_mat_path, delimiter=',')
    abs_mat = np.abs(dynamics_mat)
    flattened = abs_mat.flatten()
    sorted = np.sort(flattened)
    percentiles = np.linspace(0, 1, len(flattened))
    indices = np.searchsorted(sorted, flattened, side='left')
    percentiles_mat = percentiles[indices]
    percentiles_mat = percentiles_mat.reshape(dynamics_mat.shape)
    percentiles_mat_path = os.path.join(current_dir, output_root_dir, 'percentiles_mat.csv')
    np.savetxt(percentiles_mat_path, percentiles_mat, delimiter=',')

