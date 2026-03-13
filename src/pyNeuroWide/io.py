"""
io.py

Functions to handle loading and storing data.

author: Bradley Rauscher (March, 2026)
"""
# %%
import numpy as np
import h5py
import os

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
    return rawImage[0:fileImport['imagesRequested']].reshape(-1, n_channels, metadata['aoiheight'], metadata['aoiwidth'])

def import_settings(path: str):
    with h5py.File(path, "r") as f:
        fieldnames = list(f['settings'].keys())
        f_settings = f['settings']

        N = len(fieldnames)

        settings = {}

        for i in range(N):
            settings[fieldnames[i]] = f_settings[fieldnames[i]][:]

    return settings

# ======================================
# saving functions
# ======================================