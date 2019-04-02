#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing ECG signal.

Copyright 2017 - 2019 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, March 2017
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
import bme

# %% init
file_name = 'ecg.txt'
ecg = np.loadtxt(file_name)
fs = 1000
n_bit = 12

# %% normalise and plot
ecg_norm = ecg - 2**(n_bit-1)
ecg_norm = ecg_norm / 2**(n_bit-1)
n = ecg.size
t = np.arange(n) / fs
plt.figure()
plt.plot(t, ecg_norm)
plt.grid()

# %% spectrum
f, ecg_spec = bme.get_spectrum(fs, ecg_norm)
plt.figure()
plt.plot(f, ecg_spec)
plt.grid()

# %% spectrogram
t, f, spectrogram = bme.get_spectrogram(fs, ecg_norm, n_win=256)
bme.show_spectrogram(t, f, spectrogram)

# %% filter bandpass 3 - 45 Hz
f_l = 3
f_h = 45
order = int(0.3 * fs)
b = sig.firwin(order, (f_l, f_h), pass_zero=False, fs=fs, window='hann')
f, h_f = sig.freqz(b, fs=fs)
excite = np.zeros(300)
excite[0] = 1
h_n = sig.lfilter(b, 1, excite)

# %% plot
plt.figure()
plt.subplot(211)
plt.plot(f, 20*np.log10(np.abs(h_f)))
plt.grid()
plt.subplot(212)
plt.plot(h_n)
plt.grid()

# %% filter ecg signal
ecg_filt = sig.lfilter(b, 1, ecg_norm)
ecg_filtfilt = sig.filtfilt(b, 1, ecg_norm)
t = np.arange(n) / fs
plt.figure()
plt.plot(t, ecg_norm)
plt.plot(t, ecg_filt)
plt.plot(t, ecg_filtfilt)
plt.grid()
plt.legend(['ecg_norm', 'ecg_filt', 'ecg_filtfilt'])

# %% locate R peaks
thresh = 0.6 * np.max(ecg_filtfilt)
ecg = ecg_filtfilt.copy()
ecg[ecg < thresh] = 0

plt.figure()
plt.plot(t, ecg_filtfilt)
plt.plot(t, ecg)
plt.grid()

# %% accumulate Rs
rs = []
for i in range(1, len(ecg)-1):
    if ecg[i-1] < ecg[i] >= ecg[i+1]:
        rs.append(i)
rs = np.array(rs)
rs_t = rs / fs
rs_t_diff = np.diff(rs_t)  # s
rs_t_f = 1 / rs_t_diff  # Hz
rs_t_bpm = rs_t_f * 60

plt.figure()
plt.plot(rs_t[1:], rs_t_bpm)
plt.grid()

# moving average ...

# %% RR diagram
plt.figure()
plt.plot(rs_t_diff[:-1], rs_t_diff[1:], 'o')
minmax = (np.min(rs_t_diff), np.max(rs_t_diff))
plt.plot(minmax, minmax, ':', c='r')
plt.grid()

# %% average ecg signature
off_l = int(0.200 * fs)  # levo od R
off_r = int(0.400 * fs)  # desno od R
n_win = off_l + off_r
ecgs = np.zeros((rs.size, n_win))
for i, r in enumerate(rs):
    ecgs[i, :] = ecg_filtfilt[r-off_l : r+off_r]
ecg_signature = np.mean(ecgs, axis=0)

# %% plot
plt.figure()
plt.plot(ecgs.T)
plt.plot(ecg_signature, 'w', lw=4, alpha=.8)
plt.plot(ecg_signature, 'k', lw=2, alpha=.8)
plt.grid()
