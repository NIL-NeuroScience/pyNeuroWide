"""
utils.py

Utility functions for analyzing widefield neural datasets.

author: Bradley Rauscher (March, 2026)
"""
# %%
import numpy as np
import h5py
import os

# %%

def list_dir_struct(path: str):
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    N = len(folders)

    dir_struct = {}

    for i in range(N):
        dir_struct[folders[i]] = os.listdir(path + '/' + folders[i])
    
    return dir_struct

def list_runs(files: str):
    N = len(files)
    
    if N == 0:
        return None

    runs = []

    for i in range(N):
        filename = []
        for f in files[i]:
            if f == '.':
                break

            if f.isdigit():
                filename.append(f)
        
        if len(filename):
            runs.append(int(''.join(filename)))

    return runs