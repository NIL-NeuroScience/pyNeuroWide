"""
Multi-taper spectral estimation module.

Python translation of the Chronux mtspectrumc function.
https://chronux.org
"""
# %%
import numpy as np
from scipy.signal.windows import dpss as scipy_dpss
from scipy.stats import chi2, t
from scipy import signal

# %%
def coherencyc(data1, data2, tapers=[3,5], Fs=None, params=None, T=None):
    data1 = change_row_to_column(data1)
    data2 = change_row_to_column(data2)
    if params is None:
        params = {}
    
    if Fs is not None:
        params['Fs'] = Fs

    if tapers is not None:
        params['tapers'] = tapers
    
    tapers, pad, Fs, fpass, err, trialave = getparams(params)

    N = data1.shape[0]
    
    if T is None:
        nfft = max(2 ** (int(np.ceil(np.log2(N))) + pad), N)
    else:
        nfft = max(2 ** (int(np.ceil(np.log2(T))) + pad), T)

    f, findx = getfgrid(Fs, nfft, fpass)
    
    # Check and compute tapers
    tapers, eigs = dpsschk(tapers, N, Fs)
    
    # Compute multi-taper FFT
    J1 = mtfftc(data1, tapers, nfft, Fs)
    J2 = mtfftc(data2, tapers, nfft, Fs)
    
    # Extract frequencies of interest
    J1 = J1[findx, :, :]
    J2 = J2[findx, :, :]
    
    # Compute power spectrum (mean of the squared tapered FFTs)
    S12 = np.mean(np.conj(J1) * J2, axis=1)  # Average over tapers
    S1 = np.mean(np.conj(J1) * J1, axis=1)  # Average over tapers
    S2 = np.mean(np.conj(J2) * J2, axis=1)  # Average over tapers

    if trialave and S1.ndim > 1:
        S12 = np.mean(S12, axis=1)
        S1 = np.mean(S1, axis=1)
        S2 = np.mean(S2, axis=1)
    
    C12 = S12 / np.sqrt(S1 * S2)
    C = np.abs(C12)
    phi = np.angle(C12)

    return C, phi, f

def mtspectrumc(data, tapers=[3,5], Fs=None, params=None, T=None):
    """
    Multi-taper spectrum estimation for continuous data.
    
    Computes the multi-taper power spectrum using DPSS tapers.
    
    Parameters
    ----------
    data : ndarray
        Input data of shape (samples,) or (samples, channels/trials).
    params : dict, optional
        Dictionary with optional fields:
        - tapers : ndarray or list
            Either:
            (1) [TW, K] where TW is time-bandwidth product and K is number of tapers
            (2) [W, T, p] where W is bandwidth, T is duration, p is integer
                (2TW-p tapers used). Default is [3, 5].
        - pad : int
            FFT padding factor. -1 = no padding, 0 = pad to next power of 2, etc.
            Default is 0.
        - Fs : float
            Sampling frequency. Default is 1.
        - fpass : ndarray or list
            Frequency band [fmin, fmax] in Hz. Default is [0, Fs/2].
        - err : int or list
            Error calculation: [0] or 0 = no errors (default), 
            [1, p] = theoretical errors, [2, p] = jackknife errors.
        - trialave : bool
            If True, average over channels/trials. Default is False.
    
    Returns
    -------
    S : ndarray
        Power spectrum. Shape is (frequencies,) if trialave=True,
        else (frequencies, channels/trials).
    f : ndarray
        Frequency vector.
    Serr : ndarray, optional
        Error estimates (only if err[0] >= 1).
        Shape is (2, frequencies) or (2, frequencies, channels/trials).
    """
    if params is None:
        params = {}
    
    if Fs is not None:
        params['Fs'] = Fs

    if tapers is not None:
        params['tapers'] = tapers

    # Extract and validate parameters
    tapers, pad, Fs, fpass, err, trialave = getparams(params)
    
    # Ensure data is 2D (samples, channels)
    data = change_row_to_column(data)
    N = data.shape[0]
    
    # Compute FFT size
    if T is None:
        nfft = max(2 ** (int(np.ceil(np.log2(N))) + pad), N)
    else:
        nfft = max(2 ** (int(np.ceil(np.log2(T))) + pad), T)
    
    # Get frequency grid
    f, findx = getfgrid(Fs, nfft, fpass)
    
    # Check and compute tapers
    tapers, eigs = dpsschk(tapers, N, Fs)
    
    # Compute multi-taper FFT
    J = mtfftc(data, tapers, nfft, Fs)
    
    # Extract frequencies of interest
    J = J[findx, :, :]
    
    # Compute power spectrum (mean of the squared tapered FFTs)
    S = np.mean(np.conj(J) * J, axis=1)  # Average over tapers
    
    # Average over trials/channels if requested
    if trialave and S.ndim > 1:
        S = np.mean(S, axis=1)
    elif S.ndim == 1:
        pass  # Already 1D
    else:
        if trialave:
            S = np.mean(S, axis=1)
    
    # Compute error bars if requested
    Serr = None
    if err[0] >= 1:
        Serr = specerr(S, J, err, trialave)
    
    if Serr is not None:
        return S, f, Serr
    else:
        return np.real(S), f

def change_row_to_column(data):
    """
    Transform 1D arrays into column vectors.
    
    Parameters
    ----------
    data : ndarray
        Input array or matrix.
    
    Returns
    -------
    data : ndarray
        Data as column vector or unchanged if already 2D matrix.
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data


def getparams(params):
    """
    Extract and set default parameter values.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary.
    
    Returns
    -------
    tapers : ndarray or list
        Taper specification.
    pad : int
        FFT padding factor.
    Fs : float
        Sampling frequency.
    fpass : ndarray
        Frequency band.
    err : list
        Error calculation specification.
    trialave : bool
        Whether to average over trials.
    """
    # Default values
    if 'tapers' not in params or params['tapers'] is None:
        print('tapers unspecified, defaulting to params.tapers=[3, 5]')
        tapers = [3, 5]
    else:
        tapers = params['tapers']
    
    # Convert [W, T, p] format to [TW, K] format
    if isinstance(tapers, (list, tuple, np.ndarray)):
        tapers = np.asarray(tapers)
        if len(tapers) == 3:
            TW = tapers[1] * tapers[0]  # T * W
            K = int(np.floor(2 * TW - tapers[2]))
            tapers = [TW, K]
    
    pad = params.get('pad', 0)
    Fs = params.get('Fs', 1)
    
    if 'fpass' not in params or params['fpass'] is None:
        fpass = [0, Fs / 2]
    else:
        fpass = params['fpass']
    
    if 'err' not in params or params['err'] is None:
        err = [0, 0.05]
    else:
        err = params['err']
        if isinstance(err, (int, float)):
            err = [int(err), 0.05]
    
    trialave = params.get('trialave', False)
    
    return tapers, pad, Fs, fpass, err, trialave


def getfgrid(Fs, nfft, fpass):
    """
    Get frequency grid associated with FFT.
    
    Parameters
    ----------
    Fs : float
        Sampling frequency.
    nfft : int
        Number of FFT points.
    fpass : array-like
        Frequency band [fmin, fmax].
    
    Returns
    -------
    f : ndarray
        Frequency vector.
    findx : ndarray
        Indices of frequencies in the full grid.
    """
    df = Fs / nfft
    f_full = np.arange(0, Fs, df)
    f_full = f_full[:nfft]
    
    if np.isscalar(fpass):
        # Find closest frequency to fpass
        findx = np.argmin(np.abs(f_full - fpass))
        findx = np.array([findx])
    else:
        # Find frequencies within [fmin, fmax]
        fpass = np.asarray(fpass)
        findx = np.where((f_full >= fpass[0]) & (f_full <= fpass[-1]))[0]
    
    f = f_full[findx]
    
    return f, findx


def dpsschk(tapers, N, Fs):
    """
    Calculate or check DPSS tapers.
    
    Parameters
    ----------
    tapers : array-like
        Either precalculated tapers or [TW, K] specification.
    N : int
        Number of samples.
    Fs : float
        Sampling frequency.
    
    Returns
    -------
    tapers : ndarray
        DPSS tapers of shape (N, K).
    eigs : ndarray
        Eigenvalues of the tapers.
    """
    tapers = np.asarray(tapers)
    
    # If tapers is [TW, K], compute DPSS tapers
    if tapers.ndim == 1 and len(tapers) == 2:
        TW, K = int(tapers[0]), int(tapers[1])
        tapers_out, eigs = scipy_dpss(N, TW, K, return_ratios=True)
        # Normalize by sqrt(Fs)
        tapers_out = tapers_out * np.sqrt(Fs)
        return tapers_out.transpose(), eigs
    elif tapers.ndim == 2:
        # Precalculated tapers
        if tapers.shape[0] != N:
            raise ValueError('Length of tapers is incompatible with length of data')
        eigs = np.ones(tapers.shape[1])
        return tapers, eigs
    else:
        raise ValueError('Invalid taper specification')


def mtfftc(data, tapers, nfft, Fs):
    """
    Compute multi-taper FFT for continuous data.
    
    Parameters
    ----------
    data : ndarray
        Input data of shape (samples,) or (samples, channels).
    tapers : ndarray
        DPSS tapers of shape (N, K).
    nfft : int
        FFT size.
    Fs : float
        Sampling frequency.
    
    Returns
    -------
    J : ndarray
        FFT coefficients of shape (nfft, K, channels).
    """
    data = change_row_to_column(data)
    NC, C = data.shape  # NC = samples, C = channels
    NK, K = tapers.shape  # NK = samples, K = tapers
    
    if NK != NC:
        raise ValueError('Length of tapers is incompatible with length of data')
    
    # Reshape to (samples, tapers, channels)
    # Taper data
    tapers_expanded = tapers[:, :, np.newaxis]  # (NC, K, 1)
    data_expanded = data[:, np.newaxis, :]  # (NC, 1, C)
    
    # Apply tapers: element-wise multiply
    data_proj = data_expanded * tapers_expanded  # (NC, K, C)
    
    # Compute FFT along first dimension (samples)
    J = np.fft.fft(data_proj, n=nfft, axis=0) / Fs
    
    return J


def specerr(S, J, err, trialave, numsp=None):
    """
    Compute error bars on spectrum.
    
    Parameters
    ----------
    S : ndarray
        Power spectrum.
    J : ndarray
        Tapered FFTs of shape (nfreq, K, C).
    err : array-like
        Error specification [errtype, p].
        errtype=1: asymptotic (chi-square based)
        errtype=2: jackknife
    trialave : bool
        Whether trials were averaged.
    numsp : ndarray, optional
        Number of spikes per channel (for point process data).
    
    Returns
    -------
    Serr : ndarray
        Error estimates of shape (2, nfreq) or (2, nfreq, C).
    """
    if err[0] == 0:
        raise ValueError('Need err=[1, p] or [2, p] for error bar calculation')
    
    nf, K, C = J.shape
    errchk = int(err[0])
    p = err[1]
    pp = 1 - p / 2
    qq = 1 - pp
    
    if trialave:
        dim = K * C
        C = 1
        dof = 2 * dim
        if numsp is not None:
            dof = int(1 / (1 / dof + 1 / (2 * np.sum(numsp))))
        J = J.reshape(nf, dim)
        dof_arr = np.array([dof])
    else:
        dim = K
        dof_arr = 2 * dim * np.ones(C, dtype=int)
        if numsp is not None:
            for ch in range(C):
                dof_arr[ch] = int(1 / (1 / dof_arr[ch] + 1 / (2 * numsp[ch])))
    
    Serr = np.zeros((2, nf, C if not trialave else 1))
    
    if errchk == 1:
        # Theoretical (chi-square based) error bars
        Qp = chi2.ppf(pp, dof_arr)
        Qq = chi2.ppf(qq, dof_arr)
        
        if trialave:
            Serr[0, :, 0] = dof_arr[0] * S / Qp[0]
            Serr[1, :, 0] = dof_arr[0] * S / Qq[0]
        else:
            for ch in range(C):
                Serr[0, :, ch] = dof_arr[ch] * S[:, ch] / Qp[ch]
                Serr[1, :, ch] = dof_arr[ch] * S[:, ch] / Qq[ch]
    
    elif errchk == 2:
        # Jackknife error bars
        tcrit = t.ppf(pp, dim - 1)
        Sjk = np.zeros((dim, nf, C))
        
        for k in range(dim):
            indices = np.setdiff1d(np.arange(dim), k)
            Jjk = J[:, indices, :]  # 1-drop projection
            eJjk = np.sum(np.conj(Jjk) * Jjk, axis=1)
            Sjk[k, :, :] = eJjk / (dim - 1)
        
        sigma = np.sqrt(dim - 1) * np.std(np.log(Sjk), axis=0)
        conf = tcrit * sigma
        
        if trialave:
            Serr[0, :, 0] = S * np.exp(-conf[:, 0])
            Serr[1, :, 0] = S * np.exp(conf[:, 0])
        else:
            for ch in range(C):
                Serr[0, :, ch] = S[:, ch] * np.exp(-conf[:, ch])
                Serr[1, :, ch] = S[:, ch] * np.exp(conf[:, ch])
    
    Serr = np.squeeze(Serr)
    
    return Serr