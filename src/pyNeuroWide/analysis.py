# %%
import numpy as np
from importlib.resources import files
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt
from suite2p.run_s2p import run_s2p
from suite2p import default_ops
import os
from tifffile import imread
import tifffile
import tempfile
from pyNeuroWide import utils
from sklearn.decomposition import PCA

# %%

def pca(signal, axis=0, n=[]):
    t, x, y = signal.shape
    signal = signal.reshape(t, x * y)
    if axis == 1:
        signal = signal.transpose(1, 0)

    pca = PCA(n_components=n)

    X_pca = pca.fit_transform(signal)

    if axis == 1:
        X_pca = X_pca.reshape(x, y, n).transpose(2, 0, 1)

    return X_pca, pca