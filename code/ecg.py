#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing ECG signal.

Copyright 2017 - 2024 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, March 2024
"""
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt
import bme

ekg = np.loadtxt("data/ecg.txt")
fs = 1000  # Hz
t = np.arange(ekg.size) / fs

n_bit = 12
ekg = ekg - 2**11
ekg = ekg / 2**11

# %% plot waveform
plt.figure()
plt.plot(t, ekg)
plt.grid()

# %% plot spectrum
bme.get_spectrum(fs, ekg, plot=True)

# %% plot spectrogram
bme.get_spectrogram(fs, ekg)

# %% filter ekg
# %%% notch filter design
b, a = sig.iirnotch(50, 30, fs=fs)
f, h_fft = sig.freqz(b, a, worN=2048, fs=fs)
plt.figure()
plt.plot(f, 20 * np.log10(np.abs(h_fft)))
plt.grid()

# %%%% notch filter signal
ekg_notch = sig.lfilter(b, a, ekg)

# %%%% plot notch filtered signal
bme.get_spectrum(fs, ekg_notch, plot=True)
bme.get_spectrum(fs, ekg, plot=True)

bme.get_spectrogram(fs, ekg_notch)
bme.get_spectrogram(fs, ekg)

plt.figure()
plt.plot(t, ekg)
plt.plot(t, ekg_notch)
plt.grid()

# %%% bandpass filter
f_l = 0.5  # Hz
f_h = 75  # Hz
# b, a = sig.iirfilter(5, [f_l, f_h], fs=fs)
b = sig.firwin(3001, [f_l, f_h], pass_zero=False, fs=fs)

bme.plot_filter(b, 1, fs)

# %%% filter signal
ekg_filt = sig.convolve(ekg_notch, b, mode="same")

bme.get_spectrum(fs, ekg_filt, plot=True)
bme.get_spectrum(fs, ekg, plot=True)

bme.get_spectrogram(fs, ekg)
bme.get_spectrogram(fs, ekg_filt)

# %% plot signal
plt.figure()
plt.plot(t, ekg, alpha=0.4)
plt.plot(t, ekg_filt)
plt.grid()

# %% bpm
# %%% thresholding
thresh = 0.6 * ekg_filt.max()
ekg_thresh = ekg_filt.copy()
ekg_thresh[ekg_thresh < thresh] = 0
rs = []
for i, amp in enumerate(ekg_thresh):
    if i == 0:
        continue
    if i == ekg_thresh.size - 1:
        continue
    if ekg_thresh[i - 1] < amp >= ekg_thresh[i + 1]:
        rs.append(i)

rs = np.array(rs)

# %%% plot Rs
plt.figure()
plt.plot(t, ekg_filt)
plt.plot(t[rs], ekg_filt[rs], "ro")
plt.grid()

# %%% calculate and plot bpm
peak_final_ts = t[rs]
delta_peak_ts = np.diff(peak_final_ts)
bpm = 60 / delta_peak_ts
t_bpm = peak_final_ts[1:]

plt.figure()
plt.plot(t_bpm, bpm)
plt.grid()

# %% R-R diagram
delta_peak_ts = np.diff(peak_final_ts)
r_r_prev = delta_peak_ts[:-1]
r_r_next = delta_peak_ts[1:]

plt.figure()
plt.plot([0.2, 1.2], [0.2, 1.2], "--r")
plt.plot(r_r_prev, r_r_next, "o")
plt.grid()
