#!/usr/bin/env python3
"""
Biomedical electronics utility functions.

Copyright 2017 - 2024 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, March 2017
"""
import numpy as np
from scipy import fftpack as fft
from scipy import signal as sig
from matplotlib import pyplot as plt


def get_spectrum(fs, x, n_fft=None, log=True, plot=False):
    """
    Calculate FFT spectrum.

    Parameters
    ----------
    fs : float
        Sampling frequency.
    x : array_like
        Input signal.
    n_fft : int, optional
        FFT size.
    log : bool, optional
        Logarithmic scale.
    plot : bool, optional
        Plot spectrum. If True, the function returns None.

    Returns
    -------
    f : array_like
        Frequency axis.
    x_spec : array_like
        Spectrum.
    """
    n = x.size
    if n_fft is None:
        n_fft = int(2**np.ceil(np.log2(n)))
    x_spec = fft.fft(x, n_fft)
    x_spec = np.abs(x_spec)
    x_spec = x_spec / n
    n_keep = int(n_fft / 2) + 1
    x_spec = x_spec[0:n_keep]
    x_spec[1: -1] = 2 * x_spec[1: -1]
    f = np.linspace(0, fs / 2, n_keep)
    if log:
        x_spec = 20 * np.log10(x_spec)
    if plot:
        plt.figure()
        plt.plot(f, x_spec)
        plt.grid()
    else:
        return f, x_spec


def get_spectrogram(fs, x, n_win=None, log=True, win_type='hann', plot=True,):
    """
    Calculate spectrogram.

    Parameters
    ----------
    fs : float
        Sampling frequency.
    x : array_like
        Input signal.
    n_win : int, optional
        Window size.
    log : bool, optional
        Logarithmic scale.
    win_type : str, optional
        Window type.
    plot : bool, optional
        Plot spectrogram.

    Returns
    -------
    t : array_like
        Time axis.
    f : array_like
        Frequency axis.
    frames : array_like
        Spectrogram frames.
    """
    if n_win is None:
        n_win = 256
    win = sig.get_window(win_type, n_win)
    n_half = int(n_win / 2)
    n_hop = n_half
    pad = np.zeros(n_half)
    x_pad = np.concatenate((pad, x, pad))
    pos = 0
    frames = None
    while pos <= x_pad.size - n_win:
        frame = x_pad[pos: pos + n_win]
        frame = frame * win
        f, frame_spec = get_spectrum(fs, frame, n_win, log=log)
        frame_2d = frame_spec[:, np.newaxis]
        if frames is None:
            frames = frame_2d
        else:
            frames = np.concatenate((frames, frame_2d), axis=1)
        pos += n_hop

    t = np.arange(0, frames.shape[1] * n_hop / fs, n_hop / fs)

    if plot:
        show_spectrogram(t, f, frames)
        return
    else:
        return t, f, frames


def show_spectrogram(t, f, spectrogram, vmin=-100, vmax=None):
    """
    Show spectrogram.

    Parameters
    ----------
    t : array_like
        Time axis.
    f : array_like
        Frequency axis.
    spectrogram : array_like
        Spectrogram.
    vmin : float, optional
        Minimum value.
    vmax : float, optional
        Maximum value.
    """
    plt.figure()
    plt.imshow(
        spectrogram,
        extent=[0, t[-1], 0, f[-1]],
        aspect='auto', origin='lower',
        vmin=vmin, vmax=vmax,
        )
    plt.colorbar()


def plot_filter(b, a, fs):
    """
    Plot filter amplitude response.

    Parameters
    ----------
    b : array_like
        Numerator coefficients.
    a : array_like
        Denominator coefficients.
    fs : float
        Sampling frequency.
    """
    f, h_fft = sig.freqz(b, a, worN=2048, fs=fs)
    plt.figure()
    plt.plot(f, 20 * np.log10(np.abs(h_fft)))
    plt.grid()
