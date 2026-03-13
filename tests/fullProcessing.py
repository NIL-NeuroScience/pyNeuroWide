# %%
from pyNeuroWide import io, utils
from pyNeuroWide import processing as pnw
import numpy as np

# %% inputs
path = '/projectnb/devorlab/bcraus/HRF/1P/26-02-18/Rbp4_132'
rotation = 180

# %% organize files and settings
path_onephoton = path + '/onephoton/Run01'
path_settings = path + '/Triggers/Run001.mat'

# list file structure
file_struct = utils.list_dir_struct(path)

settings = io.import_settings(path_settings)
settings['LEDOrder'] = settings['LEDOrder'].transpose()
settings['LEDOrder'] = [''.join([chr(c) for c in f]) for f in settings['LEDOrder']]


# %% get indices for each imaging channel
rfp_idx = [i for i,v in enumerate(settings['LEDOrder']) if v == '565'][0]
gfp_idx = [i for i,v in enumerate(settings['LEDOrder']) if v == '470'][0]
HD1_idx = [i for i,v in enumerate(settings['LEDOrder']) if v == '525'][0]
HD2_idx = [i for i,v in enumerate(settings['LEDOrder']) if v == '625'][0]

# %% load imaging data from .dat file

rawData = io.import_DAT(path=path_onephoton, n_channels=4, frames=400)

# process data
n_rotations = rotation // 90

rawData = np.rot90(rawData, k=n_rotations, axes=(2,3))

HbO, Hb = pnw.estimateHemodynamics(rawData[:,HD1_idx], rawData[:,HD2_idx], 525, 625)

gfp = pnw.deltaF(rawData[:,gfp_idx])
gfp_HD = pnw.green_HD_correction(gfp, HbO, Hb)

rfp = pnw.deltaF(rawData[:,rfp_idx])
rfp_HD = pnw.red_HD_correction(rfp, rawData[:,HD1_idx], rawData[:,HD2_idx])

# %%

from matplotlib import pyplot as plt

img = gfp.std(axis=0)
plt.imshow(img)