# %%
import os
import numpy as np
import json
from pyNeuroWide import io, utils
from pathlib import Path
from scipy.io import loadmat
from fullProcessing import full_processing

# %%

path = "/projectnb/devorlab/bcraus/HRF/1P"

contents = []
for filename in os.listdir(path):
    for second_level in os.listdir(path + "/" + filename):
        contents.append(path + "/" + filename + "/" + second_level)

contents = sorted(contents)

# %% find dataIn rotation

contents = contents[::-1]
rotations = []
flag_json = []

for filename in contents:
    dataIn_path = filename + "/dataIn.mat"
    dataIn_json_path = filename + "/dataIn.json"

    if os.path.exists(dataIn_path):
        # print(f"Converting {filename}")
        dataIn = loadmat(dataIn_path, struct_as_record=False, squeeze_me=True)["dataIn"]

        if isinstance(dataIn, np.ndarray):
            dataIn = dataIn[0]

        if "rotation" in dataIn._fieldnames:
            rotations.append(dataIn.rotation)
        else:
            rotations.append(1)
    else:
        rotations.append(None)

    if os.path.exists(dataIn_json_path):
        flag_json.append(True)
    else:
        flag_json.append(False)

# %%

N = len(contents)

for i in range(30):
    if not flag_json[i] and rotations[i] and rotations[i] != 1:
        full_processing(contents[i], rotations[i])