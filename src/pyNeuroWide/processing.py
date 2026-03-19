"""
processing.py

Initial processing functions for widefield neural recordings 
including hemodynamic estimation, artifact correction, and 
deltaF/F caclulation.

author: Bradley Rauscher (March, 2026)
"""

# =================================
# Imports
# =================================
# %%
import numpy as np
from importlib.resources import files
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# %%
# =================================
# Public Functions
# =================================

def green_HD_correction(gfp, HbO, HbR, WL=np.array([470, 515])):
    """Apply a correction to the green channel signal to 
    account for HD artifact."""
    
    exs = getExtinctionCoefficients(WL)
    pathEx = estimatePathlengths(np.array([WL[0]])) / 2
    pathEm = estimatePathlengths(np.array([WL[1]])) / 2

    muaEx = (exs[0, 0] * HbO) + (exs[0, 1] * HbR)
    muaEm = (exs[1, 0] * HbO) + (exs[1, 1] * HbR)

    gfp_HD = (gfp + 1) / np.exp(-(muaEx * pathEx + muaEm * pathEm)) - 1

    return gfp_HD

def red_HD_correction(rfp, ch1, ch2, baseline=None):
    """Apply a correction to the red channel signal to 
    account for HD artifact."""
    
    HbRed = deltaF(ch2, baseline) + 1
    HbGreen = deltaF(ch1, baseline) + 1

    rfp_HD = (rfp + 1) / (HbRed**0.8 * HbGreen**0.4) - 1

    return rfp_HD

def deltaF(sig, baseline=None):
    """Calculate the delta F/F signal."""
    
    if baseline is None:
        baseline = np.mean(sig, axis=0, keepdims=True)
    else:
        baseline = np.mean(sig[baseline, :, :], axis=0, keepdims=True)
    
    dF = (sig - baseline) / baseline
    
    return dF

def getExtinctionCoefficients(Lambda):
    """Returns the extinction coefficients for
    [HbO,Hb]."""

    num_lambda = len(Lambda)
    
    data_path = files("pyNeuroWide.data") / "Hb_Lambda_Coefficients.csv"
    vLambdaHbOHb = pd.read_csv(data_path, header=None)

    vLambdaHbOHb[1] = vLambdaHbOHb[1] * 2.303
    vLambdaHbOHb[2] = vLambdaHbOHb[2] * 2.303

    exs = np.zeros((num_lambda, 2))

    f1 = interp1d(vLambdaHbOHb[:][0], vLambdaHbOHb[:][1], bounds_error=False, fill_value="extrapolate")
    f2 = interp1d(vLambdaHbOHb[:][0], vLambdaHbOHb[:][2], bounds_error=False, fill_value="extrapolate")
    
    exs[:,0] = f1(Lambda)
    exs[:,1] = f2(Lambda)

    return exs

def estimatePathlengths(Lambda):
    """Returns estimated path lengths for Lambda"""
    HbO = 60e-6
    Hb = 40e-6
    g = 0.9
    c = 3e10
    e = getExtinctionCoefficients(Lambda)

    mua = e[:,0] * HbO + e[:,1] * Hb
    mus = 150 * (Lambda / 560) ** (-2)

    z0 = 1 / ((1 - g) * mus)
    gamma = np.sqrt(c / (3 * (mua + (1 - g) * mus)))

    pathlengths = (c * z0 / (2 * gamma * np.sqrt(mua * c))) * (1 + (3 / c) * mua * gamma ** 2)

    return pathlengths

def estimateHemodynamics(ch1, ch2, lambda1=525, lambda2=625):
    """Returns estimated fluctuations in HbO and Hb concentration"""
    exs = getExtinctionCoefficients(np.array([lambda1, lambda2]))
    PLs = estimatePathlengths(np.array([lambda1, lambda2]))

    exsLambda1Hb = exs[0,1]
    exsLambda1HbO = exs[0,0]
    exsLambda2Hb = exs[1,1]
    exsLambda2HbO = exs[1,0]

    pathLambda1 = PLs[0]
    pathLambda2 = PLs[1]

    cLambda1HbO = 1 / pathLambda1 * (exsLambda2Hb / (exsLambda2HbO * exsLambda1Hb - exsLambda1HbO * exsLambda2Hb))
    cLambda2HbO = 1 / pathLambda2 * (exsLambda1Hb / (exsLambda2HbO * exsLambda1Hb - exsLambda1HbO * exsLambda2Hb))
    cLambda1Hb = 1 / pathLambda1 * (exsLambda2HbO / (exsLambda2Hb * exsLambda1HbO - exsLambda2HbO * exsLambda1Hb))
    cLambda2Hb = 1 / pathLambda2 * (exsLambda1HbO / (exsLambda2Hb * exsLambda1HbO - exsLambda2HbO * exsLambda1Hb))

    A0Hb = cLambda2Hb * np.log(np.mean(ch2, axis=0)) - cLambda1Hb * np.log(np.mean(ch1, axis=0))
    A0HbO = cLambda2HbO * np.log(np.mean(ch2, axis=0)) - cLambda1HbO * np.log(np.mean(ch1, axis=0))

    HbO = A0HbO + cLambda1HbO * np.log(ch1) - cLambda2HbO * np.log(ch2)
    HbR = A0Hb + cLambda1Hb * np.log(ch1) - cLambda2Hb * np.log(ch2)

    return HbO, HbR

def smooth_2D(video, sigma, dims=[0,0,1,1]):
    # apply only to spatial dims (H, W)
    if sigma == 0:
        return video
    
    shape = tuple(sigma if d else 0 for d in dims)
    smoothed = gaussian_filter(
        video,
        sigma=shape  # H, W, C, T
    )
    return smoothed
