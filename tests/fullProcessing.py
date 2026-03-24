# %%
from pyNeuroWide import io, utils
from pyNeuroWide import processing as pnw
import numpy as np
import json
import os
import argparse

# %% INPUTS
# path = "/projectnb/devorlab/bcraus/HRF/1P/26-02-18/test"
# rotation = 180

# %%
def full_processing(path: str, rotation: int):
    # %% organize files
    print(f"Started analysis for {path}")

    file_struct = utils.list_dir_struct(path)

    fields = ["camera", "onephoton", "Triggers", "ephys"]

    runs = {}
    run_names = {}

    for name in fields:
        if name in file_struct:
            runs[name], run_names[name] = utils.list_runs(file_struct[name])
        else:
            runs[name] = []
            run_names[name] = []

    all_runs = (set(runs["camera"]) | set(runs["onephoton"]) | set(runs["ephys"])) & set(runs["Triggers"])
    all_runs = sorted(all_runs)
    N = len(all_runs)

    print(f"Found {N} total run(s)...")

    # %% initialize dataIn and settings

    print("Organizing dataIn...")
    dataIn_single_run = {
        "runnum": [],
        "solis": {},
        "daq": {},
        "behavior": {},
        "ephys": {},
        "settings": {},
        "led": [],
        "template": [],
        "rotation": 0
    }

    dataIn = []

    entry_names = ["behavior", "solis", "daq", "ephys"]
    for i in range(N):
        dataIn.append(dataIn_single_run.copy())
        dataIn[i]["runnum"] = all_runs[i]
        dataIn[i]["rotation"] = rotation
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
        
        path_settings = dataIn[i]["daq"]["folder"] + "/" + dataIn[i]["daq"]["name"]
        settings = io.import_settings(path_settings)
        dataIn[i]["settings"] = settings.copy()

        N_LEDS = len(settings["LEDOrder"])
        led = []
        for led_idx in range(N_LEDS):
            led.append({
                "type": settings["LEDOrder"][led_idx],
                "time": settings["ExposureTimes"][led_idx],
                "power": settings["LEDPower"][led_idx]
            })
        dataIn[i]["led"] = led.copy()

    # %% start preprocessing for each run

    print("Started preprocessing...")

    for i, Run in enumerate(all_runs):
        print(f"Started preprocessing for Run {Run}...")
        if dataIn[i]["solis"]:
            path_onephoton = dataIn[i]["solis"]["folder"] + "/" + dataIn[i]["solis"]["name"]
            path_sifx = path_onephoton + "/Spooled files.sifx"
            path_ini = path_onephoton + "/acquisitionmetadata.ini"
            path_bin = path_onephoton + "/data.bin"
            path_meta = path_onephoton + "/meta.json"

            flag_sifx = os.path.exists(path_sifx)
            flag_ini = os.path.exists(path_ini)
            flag_bin = os.path.exists(path_bin)
            flag_meta = os.path.exists(path_meta)

            if flag_sifx and flag_ini:
                print("\tLoading data from .dat files")
                n_channels = len(dataIn[i]["settings"]["LEDOrder"])
                rawData = io.import_DAT(path=path_onephoton, n_channels=n_channels)

                meta = {
                    "shape": list(rawData.shape),
                    "dtype": str(rawData.dtype),
                    "order": "C",
                    "axes": [
                        "T",
                        "C",
                        "H",
                        "W"
                    ],
                    "units": "intensity",
                    "channel_order": settings["LEDOrder"],
                    "rotation": dataIn[i]["rotation"]
                }

                print(f"\tSaving metadata to {path_meta}")
                with open(path_meta, "w") as f:
                    json.dump(meta, f, indent=4)

                print(f"\tSaving binary to {path_bin}")
                rawData.tofile(path_bin)

                # make sure bin_data matches rawData
                print(f"\tLoading binary from {path_bin}")
                bin_data = io.data_1P(path_onephoton)

                check = (rawData - np.rot90(bin_data.raw_data, (360 - meta["rotation"]) / 90, axes=(2,3))).sum()
                if check == 0:
                    print("\tRaw data from .dat files matches binary from data.bin")
                else:
                    raise ValueError("Binary data from data.bin does not match loaded data from .dat files!")

                del rawData

                contents = os.listdir(path_onephoton)
                for filename in contents:
                    file_path = os.path.join(path_onephoton, filename)
                    if filename != "data.bin" and filename != "meta.json":
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            utils.rmdir(file_path)

            elif flag_bin and flag_meta:
                print(f"\tLoading binary from {path_bin}")
                bin_data = io.data_1P(path_onephoton)

            elif not flag_bin or not flag_meta:
                raise ValueError("Could not find .dat files or either data.bin or meta.json!")                
        
        dataIn[i]["template"] = bin_data.raw_data.mean(axis=0)

        if dataIn[i]["behavior"]:
            path_behavior = dataIn[i]["behavior"]["folder"] + "/" + dataIn[i]["behavior"]["name"]

            flag_behavior_mp4 = os.path.isdir(path_behavior)

            if flag_behavior_mp4:
                print("\tLoading behavior .tiff files")
                beh_camera = io.import_tiff_files(path_behavior)
                
                print("\tCompressing behavior images with lossless compression")
                io.video_compression(beh_camera, output=path_behavior + ".mp4")

                print("\tLoading compressed behavior images")
                comp_beh_camera = io.load_compressed_mp4(path=path_behavior + ".mp4")

                if np.array_equal(beh_camera, comp_beh_camera):
                    print("\tSuccessfully compressed behavior video!")

                utils.rmdir(path_behavior)

                dataIn[i]["behavior"]["name"] = path_behavior + ".mp4"
        
        print(f"\tPreprocessing finished for Run {Run}")
        
    print("Saving dataIn.json")
    with open(path + "/dataIn.json", "w") as f:
        json.dump(utils.convert_to_json_safe(dataIn), f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--rotation", type=int, default=0)
    
    args = parser.parse_args()

    full_processing(args.path, args.rotation)