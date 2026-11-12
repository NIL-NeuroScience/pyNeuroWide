# %% load .mat file
from scipy.io import loadmat
import numpy as np

path = '/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/code/datasets/disinhibition.mat'
save_path = '/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/code/datasets/disinhibition.npy'

mat = loadmat(path)
mat = mat['metadata'][0]

# %% restructure data

N = mat.shape[0]

data = []

for i in np.arange(N):
    run_data = {}
    run_data['Mouse'] = mat[i]['Mouse'][0][:]
    run_data['Date'] = mat[i]['Date'][0][:]
    run_data['Run'] = int(mat[i]['Run'][0][0])
    run_data['T'] = int(mat[i]['T'][0][0])
    run_data['Type'] = mat[i]['Type'][0][:]
    run_data['gfp'] = mat[i]['gfp'][0][:]
    run_data['brain_mask'] = ~np.isnan(mat[i]['brain_mask'])
    run_data['allen'] = mat[i]['allen']
    run_data['Pupil'] = mat[i]['Pupil'][0]
    run_data['Whisking'] = mat[i]['Whisking'][0]
    run_data['Accelerometer'] = mat[i]['Accelerometer'][0]

    parcells = {}
    parcells['gfp_HD'] = mat[i]['parcells']['gfp_HD'][0][0]
    parcells['HbT'] = mat[i]['parcells']['HbT'][0][0]
    parcells['HbO'] = mat[i]['parcells']['HbO'][0][0]
    parcells['HbR'] = mat[i]['parcells']['HbR'][0][0]

    parcells_all = {}
    parcells_all['gfp_HD'] = mat[i]['parcells_all']['gfp_HD'][0][0]
    parcells_all['HbT'] = mat[i]['parcells_all']['HbT'][0][0]
    parcells_all['HbO'] = mat[i]['parcells_all']['HbO'][0][0]
    parcells_all['HbR'] = mat[i]['parcells_all']['HbR'][0][0]

    run_data['parcells'] = parcells
    run_data['parcells_all'] = parcells_all

    if mat[i]['stim'].size:
        stim = {}
        stim['gfp_ratio'] = mat[i]['stim']['gfp_ratio'][0][0]
        stim['HbT_ratio'] = mat[i]['stim']['HbT_ratio'][0][0]
        stim['t_bef'] = int(mat[i]['stim']['t_bef'][0][0][0][0])
        stim['t_aft'] = int(mat[i]['stim']['t_aft'][0][0][0][0])
        stim['t_idx'] = mat[i]['stim']['stim_idx'][0][0][0] - 1
        run_data['stim'] = stim
    else:
        run_data['stim'] = []

    data.append(run_data)

# %% save data

np.save(save_path, data, allow_pickle = True)
# %%
