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
from pyNeuroWide import process as pnw
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
# data_path = "/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/code/datasets/disinhibition.npy"
# data = np.load(data_path, allow_pickle=True)
# gfp = [d['gfp'] for d in data]
# data_Rbp4 = data[[i for i, x in enumerate(gfp) if x == "Rbp4"]]

# run = 41
# Y = data_Rbp4[run]['parcells']['HbT'][:,2:5][:,:]
# # X = np.concatenate((data_Rbp4[run]['parcells']['gfp_HD'][:,2:5,None], np.tile(data_Rbp4[run]['Pupil'][:,None,None],(1,3,1))), axis=2)
# X = data_Rbp4[run]['parcells']['gfp_HD'][:,2:5]

# %%
def xcorr(sig1: np.ndarray, 
          sig2: np.ndarray, 
          maxlag: int | None = None,
          dim: int=0
          ) -> np.ndarray:
    """Calculate cross-correlation between sig1 and sig2 along dimension dim.
    
    Args:
        sig1: First signal of shape T (timepoints) x N (channels). np.ndarray
        sig2: Second signal of shape T (timepoints) x N (channels). np.ndarray
        maxlag: Amount to limit lag range from -maxlag to +maxlag. int
        dim: Dimension to calculate cross-correlation on (default = 0). int
    
    Returns:
        Cross-correlation values from -maxlag to +maxlag for each channel (N). np.ndarray
    """

    # subtract mean from sig1 and sig2
    sig1 = sig1 - sig1.mean(axis=dim, keepdims=True)
    sig2 = sig2 - sig2.mean(axis=dim, keepdims=True)

    # calculate max lag and find transform length
    N = sig1.shape[dim]
    maxlagDefault = N - 1
    if maxlag is None:
        mxl = maxlagDefault
    else:
        mxl = min(maxlagDefault, maxlag)

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

def IRFx1(Y: np.ndarray, 
          X: np.ndarray, 
          win: list, 
          brain_mask: list, 
          norm: bool = True, 
          initialParam: list | None = None, 
          W: int = 150
          ) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Fits the following function to inputs Y and X using least squares 
        gradient descent. Returns a single set of optimized IRF parameters.

        Y(t) = conv(X(t), IRF(t0, τ1, τ2, A, B))
        
        where
        
        IRF(t0, τ1, τ2, A, B) = A * ((t-t0) / τ1)^3 * exp(-(t-t0) / τ1)
                                + B * ((t-t0) / τ2)^3 * exp(-(t-t0) / τ2)
    
        Args:
            Y: Signal to be modeled of shape (timepoints) x N (channels). np.ndarray
            X: Predictor of shape (timepoints) x N (channels). np.ndarray
            win: Time window for IRF of form [t(0), t(end)]. t(0) cannot be less than 0. list
            brain_mask: Boolean list indicating which channels to use in fitting IRF of size N. list
            norm: Normalizes each channel of Y and X. bool
            intialParam: Initial [t0, τ1, τ2] values to use (default [1, 5, 5.3]). list
            W: Width of sliding correlation kernel to calculate model performance over time. int

        Returns:
            A tuple containing:
                r: Correlation between Y and predicted Y of shape N. np.ndarray
                IRF: Optimized IRF function of shape (win[1] - win[0] + 1). np.ndarray
                params: Least-squares fitting results along with channel weights (A, B). dict
                r_dt: Sliding correlation values. np.ndarray
        """

    if initialParam is None:
        initialParam = [1, 5, 5.3]

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

    # CHANGED: precompute the masked input FFT; timing-dependent kernels are
    # transformed inside the objective.
    X_fft = np.fft.rfft(dX, n_fft, axis=0)

    def irf_cost_func(x, win, X_mat, y):
        # CHANGED: optimize timing only and solve global A/B at each step.
        irf_pos = alpha_IRF(x[0], x[1], x[2], 1, 0, win)
        irf_neg = alpha_IRF(x[0], x[1], x[2], 0, -1, win)

        pos_fft = np.fft.rfft(irf_pos, n_fft)
        neg_fft = np.fft.rfft(irf_neg, n_fft)
        conv_pos = np.fft.irfft(
            X_mat * pos_fft[:, None],
            n_fft,
            axis=0,
        )[:T]
        conv_neg = np.fft.irfft(
            X_mat * neg_fft[:, None],
            n_fft,
            axis=0,
        )[:T]

        design = np.stack(
            [conv_pos.ravel(), conv_neg.ravel()],
            axis=1,
        )
        weights = np.linalg.lstsq(design, y.ravel(), rcond=None)[0]
        prediction = (
            conv_pos * weights[0]
            + conv_neg * weights[1]
        )
        return (y - prediction).ravel()
    
    bounds = [
        (win[0], win[1]),
        (0.01, win[1]),
        (0.01, win[1]),
    ]

    params = least_squares(irf_cost_func, 
                      [t0, tau1, tau2], 
                      args=(win, X_fft, dY), 
                      bounds=([b[0] for b in bounds], [b[1] for b in bounds]))
    
    x = params['x']

    # CHANGED: evaluate the final kernels and global weights with FFT
    # convolution, then apply the resulting IRF to all pixels.
    irf_pos = alpha_IRF(x[0], x[1], x[2], 1, 0, win)
    irf_neg = alpha_IRF(x[0], x[1], x[2], 0, -1, win)
    pos_fft = np.fft.rfft(irf_pos, n_fft)
    neg_fft = np.fft.rfft(irf_neg, n_fft)

    X_fft_full = np.fft.rfft(X, n_fft, axis=0)
    conv_pos = np.fft.irfft(
        X_fft_full * pos_fft[:, None],
        n_fft,
        axis=0,
    )[:T]
    conv_neg = np.fft.irfft(
        X_fft_full * neg_fft[:, None],
        n_fft,
        axis=0,
    )[:T]

    design = np.stack(
        [conv_pos[:, brain_mask].ravel(), conv_neg[:, brain_mask].ravel()],
        axis=1,
    )
    weights = np.linalg.lstsq(
        design,
        Y[:, brain_mask].ravel(),
        rcond=None,
    )[0]
    IRF = weights[0] * irf_pos + weights[1] * irf_neg
    params["weights"] = weights

    conv = np.fft.irfft(
        X_fft_full * np.fft.rfft(IRF, n_fft)[:, None],
        n_fft,
        axis=0,
    )[:T]
    
    r = corr(conv, Y)

    r_dt = corr_dt(conv, Y, W)

    return (r, IRF, params, r_dt)

def IRFx1_varWeights(Y: np.ndarray,
                     X: np.ndarray,
                     win: list,
                     ds: int = 1,
                     norm: bool = True,
                     initialParam: list | None = None,
                     W: int = 150
                     ) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Fit a double-alpha IRF with independently weighted channels.

    Args:
        Y: Signal to be modeled. np.ndarray
        X: Predictor signal. np.ndarray
        win: Time window for the IRF, [t(0), t(end)]. list
        ds: Downsampling factor retained for compatibility. int
        norm: Normalize each channel of Y and X. bool
        initialParam: Initial [t0, tau1, tau2] values, or None. list
        W: Width of the sliding correlation kernel. int

    Returns:
        Correlation, per-channel optimized IRF, optimization parameters, and
        sliding correlation values.
    """
    # estimate double alpha function IRF kernel to fit the equation...
    #   Y(t) = IRF x X(t)

    if initialParam is None:
        initialParam = [1, 5, 5.3]

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

    # CHANGED: retain one kernel per alpha component instead of per pixel.
    X_fft = np.fft.rfft(dX, n_fft, axis=0)
    crop_start = -win[0]
    crop_stop = crop_start + T

    def irf_cost_func(x, win, X_fft, y, n_fft):
        # CHANGED: optimize timing only; solve A/B independently per pixel
        # for every candidate timing.
        irf_pos = alpha_IRF(x[0], x[1], x[2], 1, 0, win)
        irf_neg = alpha_IRF(x[0], x[1], x[2], 0, -1, win)
        pos_fft = np.fft.rfft(irf_pos, n_fft)
        neg_fft = np.fft.rfft(irf_neg, n_fft)

        conv_pos = np.fft.irfft(
            X_fft * pos_fft[:, None],
            n_fft,
            axis=0,
        )[crop_start:crop_stop]
        conv_neg = np.fft.irfft(
            X_fft * neg_fft[:, None],
            n_fft,
            axis=0,
        )[crop_start:crop_stop]

        features = np.stack([conv_pos, conv_neg], axis=-1)
        weights = pixelLR(y, features)
        prediction = (features * weights[None, :, :]).sum(axis=2)
        return (y - prediction).ravel()
    
    # CHANGED: only the three global timing parameters are nonlinear.
    bounds = [(win[0], win[1]), (0.01, win[1]), (0.01, win[1])]
    
    params = least_squares(irf_cost_func,
                           [t0, tau1, tau2],
                           args=(win, X_fft, dY, n_fft), 
                           bounds=([b[0] for b in bounds], [b[1] for b in bounds]))

    x = params['x']

    # CHANGED: build compact final component kernels and convolve once.
    t0 = x[0]
    tau1 = x[1]
    tau2 = x[2]
    irf_pos = alpha_IRF(t0, tau1, tau2, 1, 0, win)
    irf_neg = alpha_IRF(t0, tau1, tau2, 0, -1, win)
    pos_fft = np.fft.rfft(irf_pos, n_fft)
    neg_fft = np.fft.rfft(irf_neg, n_fft)

    conv_pos = np.fft.irfft(
        X_fft * pos_fft[:, None],
        n_fft,
        axis=0,
    )[crop_start:crop_stop]
    conv_neg = np.fft.irfft(
        X_fft * neg_fft[:, None],
        n_fft,
        axis=0,
    )[crop_start:crop_stop]

    features = np.stack([conv_pos, conv_neg], axis=-1)
    weights = pixelLR(dY, features)
    prediction = (features * weights[None, :, :]).sum(axis=2)

    # CHANGED: return the fitted per-pixel IRF assembled from final weights.
    IRF = (
        irf_pos[:, None] * weights[:, 0][None, :]
        + irf_neg[:, None] * weights[:, 1][None, :]
    )

    r = corr(prediction, dY)
    r_dt = corr_dt(prediction, dY, W)

    return (r, IRF, params, r_dt)

def corr(A: np.ndarray,
         B: np.ndarray,
         dim: int = 0
         ) -> np.ndarray:
    """Calculate Pearson correlation between A and B along dimension dim.

    Args:
        A: First signal. np.ndarray
        B: Second signal with a shape compatible with A. np.ndarray
        dim: Dimension along which to calculate correlation. int

    Returns:
        Correlation values along dim. np.ndarray
    """
    A0 = A - A.mean(axis=dim, keepdims=True)
    B0 = B - B.mean(axis=dim, keepdims=True)
    corr = (A0 * B0).sum(axis=dim) / (
        np.sqrt((A0**2).sum(axis=dim)) *
        np.sqrt((B0**2).sum(axis=dim))
    )
    return corr

def corr_dt(A: np.ndarray,
            B: np.ndarray,
            W: int,
            skips: int = 1,
            dim: int = 0
            ) -> np.ndarray:
    """Calculate sliding-window Pearson correlation between A and B.

    Args:
        A: First signal. np.ndarray
        B: Second signal with the same shape as A. np.ndarray
        W: Sliding-window width in timepoints. int
        skips: Number of timepoints between consecutive windows. int
        dim: Dimension along which to calculate correlation. int

    Returns:
        Correlation values for each sliding window. np.ndarray
    """
    A0 = A - A.mean(axis=dim, keepdims=True)
    B0 = B - B.mean(axis=dim, keepdims=True)

    T, R = A.shape

    sx  = np.cumsum(A, axis=0)
    sy  = np.cumsum(B, axis=0)
    sxx = np.cumsum(A * A, axis=0)
    syy = np.cumsum(B * B, axis=0)
    sxy = np.cumsum(A * B, axis=0)

    ends = np.arange(W - 1, T, skips)
    starts = ends - W + 1

    def window_sum(cs):
        out = cs[ends].copy()
        mask = starts > 0
        out[mask] -= cs[starts[mask] - 1]
        return out

    sx  = window_sum(sx)
    sy  = window_sum(sy)
    sxx = window_sum(sxx)
    syy = window_sum(syy)
    sxy = window_sum(sxy)

    numerator = W * sxy - sx * sy
    denom = np.sqrt(
        (W * sxx - sx * sx) *
        (W * syy - sy * sy)
    )
    r = np.divide(
        numerator,
        denom,
        out=np.full_like(numerator, np.nan),
        where=denom > 0
    )
    return r
    
def pixelLR(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Fit independent linear-regression weights for each pixel.

    Args:
        Y: Target signal of shape (timepoints) x N (pixels). np.ndarray
        X: Features of shape (timepoints) x N (pixels) x features. np.ndarray

    Returns:
        Regression weights for each pixel and feature. np.ndarray
    """
    ATA = np.einsum('tpi,tpj->pij', X, X)
    ATz = np.einsum('tpi,tp->pi', X, Y)

    w = np.linalg.solve(ATA, ATz[:,:,None]).squeeze(-1)
    
    return w

def FC_dt(x: np.ndarray,
          window: int,
          step: int = 1
          ) -> tuple[np.ndarray, np.ndarray]:
    """Calculate sliding-window functional connectivity correlations.

    Args:
        x: Signal of shape (timepoints) x N (regions). np.ndarray
        window: Number of timepoints in each correlation window. int
        step: Number of timepoints between consecutive windows. int

    Returns:
        Correlation matrices and window center positions.
    """
    # Remove each region's overall mean. Correlations are unchanged, and this
    # helps reduce floating-point cancellation in the cumulative sums.
    x = x - x.mean(axis=0, keepdims=True)
    n_time, n_regions = x.shape

    sums = np.pad(np.cumsum(x, axis=0), ((1, 0), (0, 0)))
    products = np.einsum("ti,tj->ijt", x, x, optimize=True)
    cross_sums = np.pad(
        np.cumsum(products, axis=2),
        ((0, 0), (0, 0), (1, 0)),
    )

    starts = np.arange(0, n_time - window + 1, step)
    ends = starts + window

    sum_w = sums[ends] - sums[starts]  # (windows, regions)
    cross_w = cross_sums[:, :, ends] - cross_sums[:, :, starts]
    cross_w = cross_w.transpose(2, 0, 1)  # (windows, regions, regions)

    cov = cross_w - np.einsum("wi,wj->wij", sum_w, sum_w) / window

    var = np.diagonal(cov, axis1=1, axis2=2)
    denom = np.sqrt(np.maximum(var[:, :, None] * var[:, None, :], 0))
    corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
    np.clip(corr, -1, 1, out=corr)

    return corr, starts+window/2

def alpha_IRF(t0: float | np.ndarray,
              tau1: float | np.ndarray,
              tau2: float | np.ndarray,
              A: float | list,
              B: float | list,
              range: list
              ) -> np.ndarray:
    """Calculate a double-alpha IRF kernel over the requested time range.

    Args:
        t0: IRF onset time. float or np.ndarray
        tau1: Time constant of the first alpha component. float or np.ndarray
        tau2: Time constant of the second alpha component. float or np.ndarray
        A: Amplitude of the first component, scalar or per-pixel list.
        B: Amplitude of the second component, scalar or per-pixel list.
        range: Time range of the form [t(0), t(end)]. list

    Returns:
        The calculated IRF kernel. np.ndarray
    """

    tr = np.arange(range[0], range[1] + 1) - t0
    D = np.zeros_like(tr, dtype=float)
    C = np.zeros_like(tr, dtype=float)
    valid = tr >= 0
    D[valid] = (tr[valid] / tau1)**3 * np.exp(-tr[valid] / tau1)
    C[valid] = (tr[valid] / tau2)**3 * np.exp(-tr[valid] / tau2)

    if isinstance(A, list):
        return np.array(A)[None,:] * D[:,None] + np.array(B)[None,:] * C[:,None]
    else:
        return A * D + B * C
    
def alphaXDeconvolve(Y: np.ndarray,
                     X: np.ndarray,
                     win: list,
                     norm: bool = True,
                     initialParam: list | None = None,
                     W: int = 150
                     ) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Fit a per-input alpha IRF to deconvolve X and model Y.

    Args:
        Y: Signal to be modeled. np.ndarray
        X: Predictor of shape (timepoints) x N (inputs) x C (channels). np.ndarray
        win: Time window for the IRF, [t(0), t(end)]. list
        norm: Normalize each channel of Y and X. bool
        initialParam: Initial [t0, tau] values, or None. list
        W: Width of the sliding correlation kernel. int

    Returns:
        Correlation, fitted IRFs, fitted parameters, and sliding correlation
        values.
    """

    if initialParam is None:
        initialParam = [1, 5]

    t0 = initialParam[0]
    tau = initialParam[1]

    T,C,N = X.shape

    # normalize Y and X
    Y = Y - Y.mean(axis=0, keepdims=True)
    X = X - X.mean(axis=0, keepdims=True)
    if norm:
        Y = Y / np.std(Y, axis=0, keepdims=True)
        X = X / np.std(X, axis=0, keepdims=True)

    n_IRF = np.diff(win)[0] + 1
    n_fft = T + n_IRF - 1

    # CHANGED: precompute values reused by every optimizer evaluation.
    y = np.squeeze(Y)
    crop_start = -win[0]
    crop_stop = crop_start + T

    X_fft = np.fft.rfft(X, n_fft, axis=0)

    # define cost function
    def irf_cost_func(x, win, X_fft, y, n_fft):
        t = x[0:N]
        tau = x[N:2*N]

        # CHANGED: the IRF depends on variable, not pixel. Avoid C copies.
        IRF = np.stack(
            [
                alpha_IRF(t[i], tau[i], 0.5, 1, 0, win)
                for i in range(N)
            ],
            axis=1,
        )

        irf_fft = np.fft.rfft(IRF, n_fft, axis=0)
        full_pred = np.fft.irfft(
            X_fft * irf_fft[:, None, :],
            n_fft,
            axis=0,
        )
        pred = full_pred[crop_start:crop_stop]
        LR = pixelLR(y, pred)
        pred = (pred * LR[None,:,:]).sum(axis=2)

        return (y - pred).ravel()

    x_in = [t0] * N + [tau] * N

    bounds = [(win[0], win[1])] * N + [(0.01, win[1] - win[0])] * N
    bounds = ([b[0] for b in bounds], [b[1] for b in bounds])

    params = least_squares(irf_cost_func,
                            x_in,
                            args=(win, X_fft, y, n_fft),
                            bounds=bounds)

    x = params['x']

    # apply kernels to full input data
    t0 = x[0:N]
    tau = x[N:2*N]

    # CHANGED: retain one compact IRF per variable rather than per pixel.
    IRF = np.stack(
        [
            alpha_IRF(t0[i], tau[i], 0.5, 1, 0, win)
            for i in range(N)
        ],
        axis=1,
    )

    # convolve full input
    IRF_fft = np.fft.rfft(IRF, n_fft, axis=0)

    full_conv = np.fft.irfft(
        X_fft * IRF_fft[:, None, :],
        n_fft,
        axis=0,
    )
    conv = full_conv[crop_start:crop_stop]

    # estimate final pixel-specific weights from the final convolution.
    w = pixelLR(y, conv)

    outParams = {
        "w": w,
        "t0": t0,
        "tau": tau
    }
    
    pred = (conv * w[None,:,:]).sum(axis=2)
    r = corr(pred[:,:,None], Y[:,:,None])
    r_dt = corr_dt(pred, np.squeeze(Y), W)
    return (r, IRF, outParams, r_dt)

# for i in range(10):
#     r, IRF, params, r_dt = IRFx1(Y,X,[0,100],[True]*3)
