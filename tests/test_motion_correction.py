# %%
import os
from tifffile import imread
import numpy as np
import tempfile
import tifffile
from suite2p import run_s2p, default_ops #default_settings, default_db
# from pyNeuroWide import utils
from threadpoolctl import threadpool_limits
import torch
torch.set_num_threads(6)

# %%
def motion_correction(path: str, ops=None):
    # suite2P motion correction

    # organize .tiff files

    print("Loading and reformatting .tiff files")
    contents = os.listdir(path)

    tiffs = []
    for file in contents:
        parts = file.split(".")
        if len(parts) > 1 and parts[-1] == "tif":
            tiffs.append(file)

    channels = []
    for tiff in tiffs:
        parts = tiff.split("Ch")
        channels.append(int(parts[1][0]))

    unique_channels = sorted(list(set(channels)))
    n_channels = len(unique_channels)

    for c in range(len(unique_channels)):
        ch_tiffs = [tiff for tiff in tiffs if "Ch" + str(unique_channels[c]) in tiff]
        ch_tiffs = sorted(ch_tiffs)
        for rep in range(len(ch_tiffs)):
            if rep == 0:
                ch_data = imread(path + "/" + ch_tiffs[rep])
            else:
                ch_data = np.concatenate([ch_data, imread(path + "/" + ch_tiffs[rep])], axis=0)
        
        if c == 0:
            all_data = ch_data
        else:
            all_data = np.stack([all_data, ch_data], axis=3)

    T, H, W, C = all_data.shape

    all_data = all_data.transpose(0,3,1,2).reshape(T * C, H, W)
    
    print("Saving reformatted data")
    tmp_save_dir = tempfile.mkdtemp()
    
    try:
        tifffile.imwrite(tmp_save_dir + "/data_chan0.tif", all_data)
        # tifffile.imwrite(tmp_save_dir + "/data_chan1.tif", all_data[:,:,:,1])
        
        print("Starting motion correction")
        if ops is None:

            ops = default_ops()
            ops['nchannels'] = 2
            ops['functional_chan'] = 2
            ops['save_mat'] = True

            # ops = default_settings()

            # ops['fs'] = 15
            # ops['diameter'] = [2.0, 2.0]
            # ops['io']['save_mat'] = True
            # ops['registration']['smooth_sigma_time'] = 2.0

        db = {}
        db['data_path'] = [tmp_save_dir]
        db['save_path0'] = path

        # db = default_db()
        # db['data_path'] = [tmp_save_dir]
        # db['nchannels'] = 2
        # db['functional_chan'] = 2
        # db['save_path0'] = path
        # db['fast_disk'] = tmp_save_dir

        # Run Suite2p
        with threadpool_limits(limits=6, user_api='blas'):
            # run_s2p(db=db, settings=ops)
            run_s2p(ops=ops, db=db)
    finally:
        print('Deleting temporary .tif data!')
        # utils.rmdir(tmp_save_dir)

if __name__ == "__main__":
    path = "/projectnb/devorlab/bcraus/HRF/2P/26-04-13/Rbp4_139/twophoton/Run04_20x_4z_475um-213"
    motion_correction(path)