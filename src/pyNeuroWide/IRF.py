"""
IRF.py

Functions to perform IRF analysis.

author: Bradley Rauscher (June, 2026)
"""
# %%
from numpy.lib.stride_tricks import sliding_window_view
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
from scipy.optimize import minimize, least_squares
from scipy.linalg import toeplitz
from scipy.signal import convolve2d, convolve
from scipy.fft import rfft, irfft

# %%

def xcorr(sig1, sig2, maxlag: int, dim=0):
    # calculates cross-correlation between sig1 and sig2 along dimension dim

    # subtract mean from sig1 and sig2
    sig1 = sig1 - sig1.mean(axis=dim, keepdims=True)
    sig2 = sig2 - sig2.mean(axis=dim, keepdims=True)

    # calculate max lag and find transform length
    N = sig1.shape[dim]
    maxlagDefault = N - 1
    mxl = min(maxlagDefault,maxlag)

    # find transform length
    m = 2 * N
    while True:
        r = m
        for p in [2,3,5,7]:
            while r > 1 and r % p == 0:
                r = r / p
        if r == 1:
            break
        m += 1

    # calculate cross correlation
    X = np.fft.fft(sig1, n=m, axis=dim)
    Y = np.fft.fft(sig2, n=m, axis=dim)
    c1 = np.fft.ifft(X * Y.conj(), axis=dim).real

    # index cross-correlation
    r_idx = np.concat((m - mxl + np.arange(0, mxl), np.arange(0, mxl + 1)))
    r = np.take(c1, r_idx, axis=dim)

    # rescale to pearson's coefficient
    cxx0 = np.sum(sig1 * sig1, axis=dim, keepdims=True)
    cyy0 = np.sum(sig2 * sig2, axis=dim, keepdims=True)
    scaleCoeffCross = np.sqrt(cxx0 * cyy0)
    
    r = r / scaleCoeffCross

    return r

def alpha_IRF(t0, tau1, tau2, A, B, range):
    # calculates double alpha function within 

    tr = np.arange(range[0], range[1] + 1) - t0
    D = (tr / tau1)**3 * np.exp(-tr / tau1)
    D[tr < 0] = 0
    C = (tr / tau2)**3 * np.exp(-tr / tau2)
    C[tr < 0] = 0

    if isinstance(A, list):
        return np.array(A)[None,:] * D[:,None] + np.array(B)[None,:] * C[:,None]
    else:
        return A * D + B * C

def IRFx1(Y, X, win, brain_mask, ds=1, norm=True, initialParam=[1,5,5.3]):
    # estimate double alpha function IRF kernel to fit the equation...
    #   Y(t) = IRF x X(t)

    t0 = initialParam[0]
    tau1 = initialParam[1]
    tau2 = initialParam[2]

    T = Y.shape[0]

    # normalize Y and X
    if norm:
        Y = Y / np.std(Y, axis=0, keepdims=True)
        X = X / np.std(X, axis=0, keepdims=True)

    # remove unwanted pixels
    dY = Y[:,brain_mask]
    dX = X[:,brain_mask]

    P = dY.shape[1]
    
    n_IRF = np.diff(win)[0] + 1
    n_fft = T + n_IRF - 1

    Xp = np.pad(dX, ((n_IRF - 1,0), (0,0)))

    Xt = sliding_window_view(Xp, n_IRF, axis=0)
    Xt = Xt[:,:,::-1]
    Xt = Xt.transpose(1, 0, 2).reshape(T * P, n_IRF)

    Yt = dY.transpose(1, 0).reshape(-1, 1)

    # deconvolution
    A = 1
    B = -1

    IRF1 = alpha_IRF(t0, tau1, tau2, A, 0, win)
    IRF2 = alpha_IRF(t0, tau1, tau2, 0, B, win)

    convPos = Xt @ IRF1[:,None]
    convNeg = Xt @ IRF2[:,None]
    
    (A,B) = np.linalg.lstsq(np.concat((convPos.ravel()[:,None],convNeg.ravel()[:,None]), axis=1), Yt.ravel()[:,None], rcond=None)[0] 
    A = A[0]
    B = -B[0]

    def irf_cost_func(x, win, X_mat, y):
        IRF = alpha_IRF(x[0], x[1], x[2], x[3], x[4], win)
        conv_result = X_mat @ IRF[:,None]
        return (y - conv_result).ravel()
    
    bounds = [(0,win[1]),(0.01,win[1]),(0.01,win[1]),(-np.inf,np.inf),(-np.inf,np.inf)]

    params = least_squares(irf_cost_func, 
                      [t0,tau1,tau2,A,B], 
                      args=(win,Xt,Yt), 
                      bounds=([b[0] for b in bounds], [b[1] for b in bounds]))
    
    x = params['x']

    IRF = alpha_IRF(x[0],x[1],x[2],x[3],x[4],win)

    conv = convolve2d(X, IRF[:,None])[0:T]
    
    r = corr(conv, Y)
    return (r, IRF, x)

def IRFx1_varWeights(Y, X, win, ds=1, norm=True, initialParam=[1,5,5.3]):
    # estimate double alpha function IRF kernel to fit the equation...
    #   Y(t) = IRF x X(t)

    # initialize parameters
    t0 = initialParam[0]
    tau1 = initialParam[1]
    tau2 = initialParam[2]

    T = Y.shape[0]
    
    # normalize X and Y
    if norm:
        Y = Y / np.std(Y, axis=0, keepdims=True)
        X = X / np.std(X, axis=0, keepdims=True)

    # remove unwanted pixels, UPDATE IN THE FUTURE!
    dY = Y
    dX = X

    P = dY.shape[1] # get downsampled pixels
    
    # calculate parameters
    n_IRF = np.diff(win)[0] + 1
    n_fft = T + n_IRF - 1

    # estimate initial weights (A,B)
    A = [1] * P
    B = [-1] * P

    IRF1 = alpha_IRF(t0, tau1, tau2, A, [0] * P, win)
    IRF2 = alpha_IRF(t0, tau1, tau2, [0] * P, B, win)

    X_fft = np.fft.rfft(dX, n_fft, axis=0)
    Pos_fft = np.fft.rfft(IRF1, n_fft, axis=0)
    Neg_fft = np.fft.rfft(IRF2, n_fft, axis=0)

    convPos = np.fft.irfft(X_fft * Pos_fft, n_fft, axis=0)[:T]
    convNeg = np.fft.irfft(X_fft * Neg_fft, n_fft, axis=0)[:T]
    
    w = pixelLR(dY, np.stack([convPos, convNeg], axis=-1))
    A = w[:,0]
    B = -w[:,1]
    
    # define cost function
    def irf_cost_func(x, win, X_fft, y, n_fft):
        IRF = alpha_IRF(x[0], x[1], x[2], list(x[3:3+P]), list(x[3+P:3+2*P]), win)
        # conv_result = np.einsum('tpk,kp->tp', X_mat, IRF)
        Y_fft = np.fft.rfft(IRF, n_fft, axis=0)
        # return (y - conv_result).ravel()
        return (y - np.fft.irfft(X_fft * Y_fft, n_fft, axis=0)[:T]).ravel()
    
    bounds = [(0,win[1]),(0.01,win[1]),(0.01,win[1])] + [(-np.inf,np.inf)] * P * 2
    
    params = least_squares(irf_cost_func,
                           [t0,tau1,tau2] + list(A) + list(B),
                           args=(win, X_fft, dY, n_fft), 
                           bounds=([b[0] for b in bounds], [b[1] for b in bounds]))

    x = params['x']

    # apply kernels to full input data
    P = Y.shape[1]
    t0 = x[0]
    tau1 = x[1]
    tau2 = x[2]
    A = [1] * P
    B = [-1] * P

    IRF1 = alpha_IRF(t0, tau1, tau2, A, [0] * P, win)
    IRF2 = alpha_IRF(t0, tau1, tau2, [0] * P, B, win)

    # convolve full input
    X_fft = np.fft.rfft(X, n_fft, axis=0)
    Pos_fft = np.fft.rfft(IRF1, n_fft, axis=0)
    Neg_fft = np.fft.rfft(IRF2, n_fft, axis=0)

    convPos = np.fft.irfft(X_fft * Pos_fft, n_fft, axis=0)[:T]
    convNeg = np.fft.irfft(X_fft * Neg_fft, n_fft, axis=0)[:T]

    w = pixelLR(Y, np.stack([convPos, convNeg], axis=-1))
    A = w[:,0]
    B = -w[:,1]

    IRF = alpha_IRF(x[0], x[1], x[2], list(A), list(B), win)
    
    r = corr(convPos * A - convNeg * B, Y)
    return (r, IRF, params)

def corr(A, B, dim=0):
    A0 = A - A.mean(axis=dim, keepdims=True)
    B0 = B - B.mean(axis=dim, keepdims=True)
    corr = (A0 * B0).sum(axis=dim) / (
        np.sqrt((A0**2).sum(axis=dim)) *
        np.sqrt((B0**2).sum(axis=dim))
    )
    return corr

def pixelLR(Y, X):
    ATA = np.einsum('tpi,tpj->pij', X, X)
    ATz = np.einsum('tpi,tp->pi', X, Y)

    w = np.linalg.solve(ATA, ATz[:,:,None]).squeeze(-1)
    
    return w
# %%
