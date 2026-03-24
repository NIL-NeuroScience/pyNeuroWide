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
    run_names = []

    for i in range(N):
        filename = []
        for f in files[i]:
            if f == '.':
                break

            if f.isdigit():
                filename.append(f)
        
        if len(filename):
            runs.append(int(''.join(filename)))
            run_names.append(files[i])

    idx = sorted(range(len(runs)), key=lambda i: runs[i])

    return [runs[i] for i in idx], [run_names[i] for i in idx]

def rmdir(path):
    contents = os.listdir(path)
    for filename in contents:
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            rmdir(file_path)
    
    os.rmdir(path)

def convert_to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_safe(v) for v in obj]
    else:
        return obj