# %%
import os
from tifffile import imread
import numpy as np
import tempfile
import tifffile
from suite2p import run_s2p, default_ops
from pyNeuroWide import utils, io, cell_classification
from threadpoolctl import threadpool_limits
import torch
import argparse
import json

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

def behavior_processing(path: str):
    beh_video = io.import_tiff_files(path)

    io.video_compression(beh_video, path + ".mp4")

    beh_video_comp = io.load_compressed_mp4(path + ".mp4")

    if np.array_equal(beh_video, beh_video_comp):
        print("\tSuccessfully compressed behavior video!")
        utils.rmdir(path)
    else:
        raise ValueError("Compressed video does not match input")
    
    return beh_video[round(beh_video.shape[0] // 2)]

def create_dataIn(path):
    file_struct = utils.list_dir_struct(path)

    fields = ["camera", "twophoton", "ephys"]

    runs = {}
    run_names = {}

    for name in fields:
        if name in file_struct:
            runs[name], run_names[name] = utils.list_runs(file_struct[name])
        else:
            runs[name] = []
            run_names[name] = []

    all_runs = set(runs["camera"]) | set(runs["twophoton"]) | set(runs["ephys"])
    all_runs = sorted(all_runs)
    N = len(all_runs)

    print(f"Found {N} total run(s)...")

    dataIn_single_run = {
        "runnum": [],
        "twophoton": {},
        "behavior": {},
        "ephys": {},
        "settings": {},
        "template": [],
        "behavior_template": []
    }

    dataIn = []

    entry_names = ["behavior", "twophoton", "ephys"]
    for i in range(N):
        dataIn.append(dataIn_single_run.copy())
        dataIn[i]["runnum"] = all_runs[i]
        dataIn[i]["ephys"] = None # for future applications
        for f_idx, field in enumerate(fields):
            if all_runs[i] in runs[field]:
                dataIn[i][entry_names[f_idx]] = {
                    "runnum": all_runs[i],
                    "name": run_names[field][runs[field].index(all_runs[i])],
                    "folder": path + "/" + field
                }
            else:
                dataIn[i][entry_names[f_idx]] = None
        
        if all_runs[i] in runs["twophoton"]:
            path_settings = dataIn[i]["twophoton"]["folder"] + "/" + dataIn[i]["twophoton"]["name"]
            contents = os.listdir(path_settings)
            path_settings = path_settings + "/" + [x for x in contents if ".xml" in x][0]
            settings = io.read_XML(path_settings)
            dataIn[i]["settings"] = settings.copy()

    return dataIn

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    # path = "/projectnb/devorlab/bcraus/HRF/2P/26-05-06/Rbp4_140"
    args = parser.parse_args()

    path = args.path
    path_twophoton = path + "/twophoton"
    path_behavior = path + "/camera"
    path_ephys = path + "/ephys"

    dataIn = create_dataIn(path)

    for run in dataIn:
        if run["twophoton"] != None:
            if not os.path.isdir(run["twophoton"]["folder"] + "/" + run["twophoton"]["name"] + "/suite2p"):
                motion_correction(run["twophoton"]["folder"] + "/" + run["twophoton"]["name"])
            
            path_suite2p = run["twophoton"]["folder"] + "/" + run["twophoton"]["name"] + "/suite2p/plane0"
            contents = os.listdir(path_suite2p)

            channels = []
            for i in contents:
                if "data" in i:
                    channels.append(i)

            channels = sorted(channels)

            ops = np.load(path_suite2p + "/ops.npy", allow_pickle=True)

            templates = []
            for c in channels:
                prefix = c.split("data")[1].split(".bin")[0]
                templates.append(ops.item()["meanImg" + prefix])
            
            run["template"] = np.stack(templates, axis=0)
        
        if run["behavior"] != None:
            if ".mp4" not in run["behavior"]["name"]:
                run["behavior_template"] = behavior_processing(run["behavior"]["folder"] + "/" + run["behavior"]["name"])
                run["behavior"]["name"] = run["behavior"]["name"] + ".mp4"
            else:
                beh_video = io.load_compressed_mp4(run["behavior"]["folder"] + "/" + run["behavior"]["name"])
                run["behavior_template"] = beh_video[round(beh_video.shape[0] // 2)]

    # save json

    with open(path + "/dataIn.json", "w") as f:
        json.dump(utils.convert_to_json_safe(dataIn), f, indent=2)