#!/usr/bin/env python3
"""
Processing EMG signal.

Copyright 2017 - 2024 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, March 2017
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
import bme

emg = np.loadtxt("data/emg.txt")
fs = 1000  # Hz
t = np.arange(emg.size) / fs  # s
n_bit = 12
emg = emg - 2**(n_bit - 1)
emg = emg / 2**(n_bit - 1)

# %% plot signal
plt.figure()
plt.plot(t, emg)
plt.grid()

# %% plot spectrum
f, spec = bme.get_spectrum(fs, emg)
plt.figure()
plt.plot(f, spec)
plt.grid()

bme.get_spectrogram(fs, emg, n_win=256)

# %% eliminate DC offset
emg = emg - emg.mean()
bme.get_spectrogram(fs, emg, n_win=256)

# %% eliminate 200 Hz harmonic of hum
f_c = 200  # Hz
for Q in [15, 30, 60]:
    b, a = sig.iirnotch(f_c, Q, fs=fs)
    f, h_fft = sig.freqz(b, a, fs=fs)
    plt.plot(f, 20 * np.log10(np.abs(h_fft)), label=str(Q))
plt.grid()
plt.legend()

# %% notch filter signal
Q = 60
b, a = sig.iirnotch(f_c, Q, fs=fs)
emg_notch = sig.lfilter(b, a, emg)

bme.get_spectrogram(fs, emg_notch, n_win=256)

# %% band pass filter signal
b, a = sig.iirfilter(9, [20, 280], btype="bandpass", ftype="butter", fs=fs)
emg_filt = sig.lfilter(b, a, emg_notch)
bme.get_spectrogram(fs, emg_filt, n_win=256)

plt.figure()
plt.plot(t, emg_filt)
plt.plot(t, emg)
plt.grid()

# %% full-wave rectifier
emg_rect = np.abs(emg_filt)

# %% moving average - convolution based
t_win = 0.050  # ms
n_win = int(t_win * fs)
win = sig.get_window("boxcar", n_win)
win = win / n_win
emg_mavg = sig.convolve(emg_rect, win, mode="same")

plt.figure()
plt.plot(t, emg)
plt.plot(t, emg_rect)
plt.plot(t, emg_mavg)
plt.grid()

# %% different t_wins
plt.figure()
for t_win in [0.500, 1, 2]:
    n_win = int(t_win * fs)
    win = sig.get_window("boxcar", n_win)
    win = win / n_win
    emg_mavg = sig.convolve(emg_rect, win, mode="same")
    plt.plot(t, emg_mavg, label=t_win, alpha=0.8)

plt.grid()
plt.legend()

# %% threshold
thresh = 1.2 * emg_mavg.mean() + 2 * emg_mavg.std()

plt.figure()
plt.plot(t, emg_mavg)
plt.plot([0, t[-1]], [thresh, thresh], "r--")
plt.grid()
plt.legend()

# %% detect onsets
onsets = []
above_thresh = False
for t_amp, emg_amp in zip(t, emg_mavg):
    if emg_amp > thresh and not above_thresh:
        onsets.append(t_amp)
        above_thresh = True
    elif emg_amp < thresh:
        above_thresh = False

# %% plot
plt.figure()
plt.plot(t, emg, alpha=0.3)
plt.plot(t, emg_mavg)
plt.plot([0, t[-1]], [thresh, thresh], "r--")
for onset in onsets:
    plt.plot([onset, onset], [-np.max(emg), np.max(emg)], lw=4, alpha=0.5)
plt.grid()
