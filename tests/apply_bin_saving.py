# %%
import os
import numpy as np
import json
from pyNeuroWide import io, utils
from pathlib import Path
from scipy.io import loadmat
from examples.fullProcessing import full_processing
from resave_behavior import resave_behCam

# %%
import os

path = "/projectnb/devorlab/bcraus/HRF/1P"

contents = []
for filename in os.listdir(path):
    for second_level in os.listdir(path + "/" + filename):
        contents.append(path + "/" + filename + "/" + second_level)

contents = sorted(contents)
contents = contents[::-1]

# %% check for mp4 files

run_paths = []
for filename in contents:
    if os.path.isdir(filename + "/camera"):
        runs = os.listdir(filename + "/camera")
    else:
        continue

    for run in runs:
        if os.path.isdir(filename + "/camera/" + run) and not os.path.isfile(filename + "/camera/" + run + "_flag.txt"):
            run_paths.append(filename + "/camera/" + run)

# %% 

def delete_cam(path):
    if not os.path.isdir(path):
        raise ValueError("Given path is not a directory!")
    
    beh_video = io.import_tiff_files(path)

    io.video_compression(beh_video, path + ".mp4")

    beh_video_comp = io.load_compressed_mp4(path + ".mp4")

    if np.array_equal(beh_video, beh_video_comp):
        print("\tSuccessfully compressed behavior video!")
    else:
        raise ValueError("Compressed video does not match input")
    
    Path.touch(path + "_flag.txt")

# %%

N = len(run_paths)

for i in range(N):
    delete_cam(run_paths[i])