"""
utils.py

Utility functions for analyzing widefield neural datasets.

author: Bradley Rauscher (March, 2026)
"""

# =================================
# Imports
# =================================
# %%
import numpy as np
import h5py

# %%
# =================================
# Public Functions
# =================================

def load_H5(path, var=None, frames=None):
    """
    Load .h5 files storing widefield imaging data.
    """
    
    if var is None:
        var = [1, 1, 1, 1, 1, 1]
    
    varPath = ['rfp/norm', 'rfp/normHD', 'gfp/norm', 'gfp/normHD', 
               'hemodynamics/HbO', 'hemodynamics/Hb']
    varName = ['rpf', 'rfp_HD', 'gfp', 'gfp_HD', 'HbO', 'HbR']

    with h5py.File(path, "r") as f:
        data = {}

        for i in range(6):
            if var[i]:
                if frames is None:
                    data[varName[i]] = f[varPath[i]][:]
                else:
                    data[varName[i]] = f[varPath[i]][frames]

    if var[4] and var[5]:
        data['HbT'] = data['HbO'] + data['HbR']

    return data