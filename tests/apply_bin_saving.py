# %%
import os
import numpy as np
import json
from pyNeuroWide import io, utils
from pathlib import Path
from scipy.io import loadmat
from fullProcessing import full_processing
from resave_behavior import resave_behCam

# %%

path = "/projectnb/devorlab/bcraus/HRF/1P"

contents = []
for filename in os.listdir(path):
    for second_level in os.listdir(path + "/" + filename):
        contents.append(path + "/" + filename + "/" + second_level)

contents = sorted(contents)
contents = contents[::-1]

# %% 

def delete_cam(path):
    if not os.path.isdir(path + "/camera"):
        return
    
    contents = os.listdir(path + "/camera")
    mp4s = [name for name in contents if ".mp4" in name]
    correct = [name for name in mp4s if "correct" in name]
    incorrect = [name for name in mp4s if "correct" not in name]

    runs_incorrect = [int(name[3:5]) for name in incorrect]
    runs_correct = [int(name[3:5]) for name in correct]

    # if set(runs_incorrect) != set(runs_correct):
    #     raise ValueError("Correct and incorrect runs do not match!!!")

    is_flag = os.path.exists(path + "/camera/flag_done.txt")
    if not is_flag:
        return
    
    # if not len(runs_correct):
    #     # os.remove(path + "/camera/flag_done.txt")
    #     return

    for runs in correct:
        new_name = runs.replace("_correct","")
        os.rename(path + "/camera/" + runs, path + "/camera/" + new_name)
    
    os.remove(path + "/camera/flag_done.txt")
    
# delete_cam(contents[0])

# %%

N = len(contents)

for i in range(N):
    # if not flag_json[i] and rotations[i] and rotations[i] != 1:
    #     print('Needs processing!')
        # full_processing(contents[i], rotations[i])
    delete_cam(contents[i])