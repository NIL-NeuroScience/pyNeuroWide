# %%
import os
from pyNeuroWide import utils

# %%
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
        if os.path.isdir(filename + "/camera/" + run) and os.path.isfile(filename + "/camera/" + run + "_flag.txt"):
            run_paths.append(filename + "/camera/" + run)

# %%

N = len(run_paths)

for i in range(N):
    utils.rmdir(run_paths[i])