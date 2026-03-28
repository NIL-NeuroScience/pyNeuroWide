# %%
from pyNeuroWide import io
import os
import argparse
from pathlib import Path

# %%
def resave_behCam(path: str):
    # %% organize files
    print(f"Started analysis for {path}")

    if not os.path.isdir(path + "/camera"):
        return
    
    contents = os.listdir(path + "/camera")

    # check files for correctly saved videos
    mp4files = [name for name in contents if ".mp4" in name]
    correct = [name for name in mp4files if "correct" in name]
    incorrect = [name for name in mp4files if "correct" not in name]
    flag = [name for name in contents if "flag" in name]
    runs = [name[0:5] for name in incorrect]

    # %%

    if len(flag):
        return
    
    for run in runs:
        print("\tLoading behavior .tiff files")
        beh_camera = io.load_compressed_mp4(path + "/camera/" + run + ".mp4")
    
        print("\tCompressing behavior images with lossless compression")
        io.video_compression(beh_camera, output=path + "/camera/" + run + "_correct.mp4")

    flag_path = Path(path + '/camera/flag_done.txt')
    flag_path.touch()


if __name__ == "__main__":
    1