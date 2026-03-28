"""
io.py

Functions to handle loading and storing data.

author: Bradley Rauscher (March, 2026)
"""
# %%
import numpy as np
import h5py

import os
import json
from pyNeuroWide import processing as pnw
from importlib.resources import files
import yaml
import imageio.v2 as imageio
import subprocess
from pathlib import Path

# %%

# ======================================
# loading functions
# ======================================
def load_H5(path, var=None, frames=None):
    """
    Load .h5 files storing widefield imaging data.
    """
    
    if var is None:
        var = [1, 1, 1, 1, 1, 1]
    
    varPath = ['rfp/norm', 'rfp/normHD', 'gfp/norm', 'gfp/normHD', 
               'hemodynamics/HbO', 'hemodynamics/Hb']
    varName = ['rpf', 'rfp_HD', 'gfp', 'gfp_HD', 'HbO', 'HbR']

    with h5py.File(path, "r") as f:
        data = {}

        for i in range(6):
            if var[i]:
                if frames is None:
                    data[varName[i]] = f[varPath[i]][:]
                else:
                    data[varName[i]] = f[varPath[i]][frames]

    if var[4] and var[5]:
        data['HbT'] = data['HbO'] + data['HbR']

    return data

def import_ini(path: str):
    """
    Parse a simple INI-like file into a dictionary.
    Lines starting with ';' or '#' or section headers [..] are ignored.
    Values that can be converted to float are converted.

    modified from https://www.mathworks.com/matlabcentral/fileexchange/17177-ini2struct
    """
    out = {}

    def clean_value(s: str):
        # modified from https://www.mathworks.com/matlabcentral/fileexchange/17177-ini2struct
        s = s.strip()
        if s.startswith('='):
            s = s[1:].strip()
        # Try converting to float
        try:
            num = int(s)
            return num
        except ValueError:
            return s

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(';') or s.startswith('#'):
                continue
            if '[' in s and ']' in s:
                continue

            if '=' in s:
                par, val = s.split('=', 1)  # split at first '=' only
                par = par.strip().lower()   # lowercase like MATLAB
                val = clean_value(val)
                # Convert par into a valid Python identifier (replace spaces, dots, etc.)
                par = par.replace(' ', '_').replace('.', '_')
                out[par] = val

    return out

def import_sifx(path: str):

    nImages = None
    acquisitionSettings = {}

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Extract number of images
            if 'Pixel number' in s:
                tmpOut = s.split()
                if len(tmpOut) == 10:
                    try:
                        nImages = int(tmpOut[6])  # MATLAB uses 1-based index
                    except ValueError:
                        pass

            # Extract PreAmpGainText
            elif '<PreAmpGainText>' in s:
                tmpOut = s.replace('>', '<').split('<')
                if len(tmpOut) == 5:
                    acquisitionSettings['preAmpSetting'] = tmpOut[2]

            # Extract ExtendedDynamicRange
            elif '<ExtendedDynamicRange>' in s:
                tmpOut = s.replace('>', '<').split('<')
                if len(tmpOut) == 5:
                    acquisitionSettings['extendedDynamicRange'] = float(tmpOut[2])

    return nImages, acquisitionSettings

def import_DAT(path: str, n_channels: int = 1, frames=None):
    # check and load metadata and sifx files
    path_meta = path + '/acquisitionmetadata.ini'
    path_sifx = path + '/Spooled files.sifx'

    if os.path.exists(path_meta) and os.path.exists(path_sifx):
        print('Found acquisitionmetadata.ini and Spooled files.sifx!')
    else:
        print('Could not find acquisitionmetadata.ini and Spooled files.sifx')
    
    metadata = import_ini(path_meta)
    metadata['totalImages_sifx'], metadata['acquisitionSettings'] = import_sifx(path_sifx)

    # adjust number of frames if necessary
    if frames:
        metadata['totalImages_sifx'] = frames
    
    if metadata['pixelencoding'] == 'Mono32':
        metadata['aoistrideFactor'] = 4
        metadata['conversionSetting'] = 'uint32=>uint32'
        metadata['rawPixelFormat'] = 'uint32'
    elif metadata['pixelencoding'] == 'Mono12':
        metadata['aoistrideFactor'] = 2
        metadata['conversionSetting'] = 'uint16=>uint16'
        metadata['rawPixelFormat'] = 'uint16'
    elif metadata['pixelencoding'] == 'Mono16':
        metadata['aoistrideFactor'] = 2
        metadata['conversionSetting'] = 'uint16=>uint16'
        metadata['rawPixelFormat'] = 'uint16'
    else:
        raise ValueError('Error! Unknown pixel encoding!')

    metadata['aoiDataSize'] = int(1 / metadata['aoistrideFactor'] * metadata['imagesizebytes'] * metadata['imagesperfile'])

    # sort .dat files
    contents = os.listdir(path)

    contents = [f for f in contents
                if f.endswith('spool.dat')]
    
    fileImport = {}
    fileImport['totalFiles'] = len(contents)
    fileImport['totalImages'] = len(contents) * metadata['imagesperfile']

    # sort files
    contents_idx = []
    for i in range(fileImport['totalFiles']):
        num = ''.join([c for c in contents[i] if c.isdigit()])
        num = int('1' + num)

        reverse = []
        while num // 10:
            reverse.append(num % 10)
            num //= 10
        
        contents_idx.append(int(''.join(map(str, reverse))))
    
    contents_idx = [i for i, _ in sorted(enumerate(contents_idx), key=lambda x: x[1])]
    contents = [contents[i] for i in contents_idx]

    # set up import
    print(f'Images acquired: {metadata['totalImages_sifx']}; Images stored: {fileImport['totalImages']}')
    print(f'Image dimensions: W {metadata['aoiwidth']} x H {metadata['aoiheight']} x T {metadata['totalImages_sifx']} ({metadata['rawPixelFormat']} depth at {metadata['pixelencoding']} encoding).')
    print(f'Image size: ~{metadata['imagesizebytes'] * fileImport['totalImages'] / 1024**3} GB')

    fileImport['imagesRequested'] = metadata['totalImages_sifx']
    fileImport['filesRequested'] = (metadata['totalImages_sifx'] + metadata['imagesperfile'] - 1) // metadata['imagesperfile']
    fileImport['imagesImported'] = fileImport['totalImages']

    # import data
    removeRows = metadata['aoiheight'] * metadata['aoiwidth'] * 2
    removeRows = metadata['imagesizebytes'] - removeRows

    print('Importing data...')
    print([fileImport['filesRequested'], metadata['aoiwidth'], metadata['aoiheight'], metadata['imagesperfile']])
    rawImage = np.zeros([fileImport['filesRequested'], metadata['imagesperfile'], metadata['aoiheight'], metadata['aoiwidth']],
                        dtype=metadata['rawPixelFormat'])

    # return metadata
    for iFile in range(fileImport['filesRequested']):
        with open(path + '/' + contents[iFile], 'r') as f:
            tmpIn = np.fromfile(f, dtype=np.dtype(metadata['rawPixelFormat']).newbyteorder('<'),
                                count=metadata['imagesizebytes'] * metadata['imagesperfile']
                                )
            metaInd = range(metadata['aoiDataSize'] // metadata['imagesperfile'], metadata['aoiDataSize'] + 1, metadata['aoiDataSize'] // metadata['imagesperfile'])
            tmpMetaInd = np.ones(metadata['aoiDataSize'], dtype=bool)
            for i in metaInd:
                tmpMetaInd[(i - 1 - removeRows // 2):(i - 1)] = 0

            rawImage[iFile] = tmpIn[tmpMetaInd].reshape((metadata['imagesperfile'], (metadata['imagesizebytes'] - removeRows) // metadata['aoistride'], metadata['aoistride'] // metadata['aoistrideFactor']))

    rawImage = rawImage.reshape(-1, metadata['aoiheight'], metadata['aoiwidth'])
    
    rawImage = rawImage[0:fileImport['imagesRequested']]
    # check if there's a mismatch in the number of frames, and add more to the beginning if necessary
    error_frames = rawImage.shape[0] % n_channels
    print(f"Number of error frames: {error_frames}")
    if error_frames != 0:
        print("Error in number of frames acquired")
        add_frames = n_channels - error_frames
        frames = np.zeros((add_frames, rawImage.shape[1], rawImage.shape[2]))
        for n in range(add_frames):
            first_idx = n_channels - add_frames + n
            frames[n] = rawImage[np.arange(first_idx, rawImage.shape[0], n_channels)].mean(axis=0)
        
        print(frames.shape)
        rawImage = np.concatenate((frames, rawImage), axis=0)
        print(rawImage.shape)

    return rawImage.reshape(-1, n_channels, metadata['aoiheight'], metadata['aoiwidth'])

def import_tiff_files(path: str):
    # properly index tiff files
    idx = []
    contents = os.listdir(path)

    for file in contents:
        idx.append(int(Path(file.split("_")[-1]).stem))
    
    idx = sorted(range(len(idx)), key=lambda i: idx[i])
    contents = [contents[i] for i in idx]
    
    first = imageio.imread(path + "/" + contents[0])
    T = len(contents)
    H, W = first.shape

    data = np.zeros((T, H, W), dtype=first.dtype)
    for i, f in enumerate(contents):
        data[i] = imageio.imread(path + "/" + f)
    return data

def video_compression(data, output: str):
    T, H, W = data.shape
    data = np.ascontiguousarray(data)

    proc = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{W}x{H}",
        "-r", "10",
        "-i", "-",
        "-c:v", "libx264",
        "-crf", "0",
        "-preset", "veryslow",
        f"{output}"
    ], stdin=subprocess.PIPE)
    
    proc.stdin.write(data.tobytes())
    proc.stdin.close()
    proc.wait()

def load_compressed_mp4(path: str):
    reader = imageio.get_reader(path)
    frames = []

    for frame in reader:
        frames.append(frame)
    data = np.stack(frames)[:,:,:,0]
    reader.close()

    return data

def import_settings(path: str):
    with h5py.File(path, "r") as f:
        fieldnames = list(f['settings'].keys())
        f_settings = f['settings']

        N = len(fieldnames)

        settings = {}

        for i in range(N):
            settings[fieldnames[i]] = f_settings[fieldnames[i]][:]

    settings["LEDOrder"] = settings["LEDOrder"].transpose()
    settings["LEDOrder"] = [int(''.join([chr(c) for c in f])) for f in settings["LEDOrder"]]
    
    return settings

# ======================================
# saving functions
# ======================================

# ======================================
# data class
# ======================================

# path = '/projectnb/devorlab/bcraus/AnalysisCode/tests'
class data_1P:
    def __init__(self, path, frames=None, smoothing=None):
        # organize paths
        self.path = path
        bin_path = path + '/data.bin'
        meta_path = path + '/meta.json'

        # intialize properties
        self._gfp = None
        self._rfp = None
        self._gfp_HD = None
        self._rfp_HD = None
        self._HbO = None
        self._HbR = None

        # setup preprocessing parameters
        config_path = files("pyNeuroWide.config").joinpath("default.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if smoothing is None:
            self.sm = config['preprocessing']['smoothing_sigma']
        else:
            self.sm = smoothing

        # load metadata from meta.json
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        
        # load data from data.bin
        if frames is None:
            raw_data = np.fromfile(bin_path, dtype=self.meta['dtype'])

            # determine channel indices
            self.raw_data = raw_data.reshape(tuple(self.meta['shape']), order=self.meta['order'])
        else:
            raw_data = np.memmap(bin_path,
                                 dtype=self.meta['dtype'],
                                 mode='r',
                                 shape=tuple(self.meta['shape']),
                                 order=self.meta['order'])
            self.raw_data = raw_data[frames].copy()

        if self.meta['rotation']:
            self.raw_data = np.rot90(self.raw_data, k=self.meta['rotation'] / 90, axes=(2,3)).copy()

    @property
    def gfp(self):
        if 470 not in self.meta['channel_order']:
            raise ValueError("Channel 470 not found! Cannot compute gfp")
        else:
            channel_idx = self.meta['channel_order'].index(470)

        if self._gfp is None:
            self._gfp = self.compute_dff(self.raw_data[:,channel_idx])
            self._gfp = pnw.smooth_2D(self._gfp, sigma=self.sm, dims=[0,1,1])
        return self._gfp
    
    @property
    def gfp_HD(self):
        if 470 not in self.meta['channel_order']:
            raise ValueError("Channel 470 not found! Cannot compute gfp")
        else:
            channel_idx = self.meta['channel_order'].index(470)

        if self._gfp_HD is None:
            gfp = self.compute_dff(self.raw_data[:,channel_idx])
            self._gfp_HD = pnw.green_HD_correction(gfp, self.HbO, self.HbR)
            self._gfp_HD = pnw.smooth_2D(self._gfp_HD, sigma=self.sm, dims=[0,1,1])

        return self._gfp_HD

    @property
    def rfp(self):
        if 565 not in self.meta['channel_order']:
            raise ValueError("Channel 565 not found! Cannot compute rfp")
        else:
            channel_idx = self.meta['channel_order'].index(565)
        
        if self._rfp is None:
            self._rfp = self.compute_dff(self.raw_data[:,channel_idx])
            self._rfp = pnw.smooth_2D(self._rfp, sigma=self.sm, dims=[0,1,1])
        return self._rfp
    
    @property
    def rfp_HD(self):
        if 565 not in self.meta['channel_order']:
            raise ValueError("Channel 565 not found! Cannot compute rfp")
        else:
            channel_idx = self.meta['channel_order'].index(565)

        if self._rfp_HD is None:
            HD1_idx = self.meta['channel_order'].index(525)
            HD2_idx = self.meta['channel_order'].index(625)

            rfp = self.compute_dff(self.raw_data[:,channel_idx])
            self._rfp_HD = pnw.red_HD_correction(rfp, self.raw_data[:,HD1_idx], self.raw_data[:,HD2_idx])
            self._rfp_HD = pnw.smooth_2D(self._rfp_HD, sigma=self.sm, dims=[0,1,1])
            
        return self._rfp_HD
    
    @property
    def HbO(self):
        if self._HbO is None:
            if 525 not in self.meta['channel_order'] and 625 not in self.meta['channel_order']:
                raise ValueError("Channel 525 and 625 not found! Cannot compute hemodynamics")
            elif 525 not in self.meta['channel_order']:
                raise ValueError("Channel 525 not found! Cannot compute hemodynamics")
            elif 625 not in self.meta['channel_order']:
                raise ValueError("Channel 625 not found! Cannot compute hemodynamics")
            else:
                HD1_idx = self.meta['channel_order'].index(525)
                HD2_idx = self.meta['channel_order'].index(625)

            self._HbO, self._HbR = pnw.estimateHemodynamics(self.raw_data[:,HD1_idx], self.raw_data[:,HD2_idx])
            self._HbO = pnw.smooth_2D(self._HbO, sigma=self.sm, dims=[0,1,1])
            self._HbR = pnw.smooth_2D(self._HbR, sigma=self.sm, dims=[0,1,1])
        return self._HbO
    
    @property
    def HbR(self):
        if self._HbR is None:
            if 525 not in self.meta['channel_order'] and 625 not in self.meta['channel_order']:
                raise ValueError("Channel 525 and 625 not found! Cannot compute hemodynamics")
            elif 525 not in self.meta['channel_order']:
                raise ValueError("Channel 525 not found! Cannot compute hemodynamics")
            elif 625 not in self.meta['channel_order']:
                raise ValueError("Channel 625 not found! Cannot compute hemodynamics")
            else:
                HD1_idx = self.meta['channel_order'].index(525)
                HD2_idx = self.meta['channel_order'].index(625)

            self._HbO, self._HbR = pnw.estimateHemodynamics(self.raw_data[:,HD1_idx], self.raw_data[:,HD2_idx])
            self._HbO = pnw.smooth_2D(self._HbO, sigma=self.sm, dims=[0,1,1])
            self._HbR = pnw.smooth_2D(self._HbR, sigma=self.sm, dims=[0,1,1])
        return self._HbR
    
    @property
    def HbT(self):
        return self.HbO + self.HbR

    def compute_dff(self, data):
        F0 = data.mean(axis=0, keepdims=1)
        return (data - F0) / F0
