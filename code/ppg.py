#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photopletismograph signal processing.

Copyright 2023 - 2024 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, Feb 2023
"""
from scipy import signal as sig
import numpy as np
from matplotlib import pyplot as plt
import bme

ppg = np.loadtxt("data/ppg.txt")
n_bit = 12
fs = 1000
t = np.arange(0, ppg.size) / fs

# scale
ppg = ppg - 2**(n_bit - 1)
ppg = ppg / 2**(n_bit - 1)

# %% plot
plt.figure()
plt.plot(t, ppg)
plt.grid()

# %% spectrum
f, spec = bme.get_spectrum(fs, ppg)

plt.figure()
plt.plot(f, spec)
plt.grid()

# %% spectrogram
ppg_norm = ppg / np.max(np.abs(ppg))
bme.get_spectrogram(fs, ppg_norm, n_win=256)

# %% filter
f_l = 25
order = 9
b, a = sig.iirfilter(order, f_l, btype="lowpass", ftype="butter", fs=fs)
f, h_fft = sig.freqz(b, a, fs=fs)

# amplitude response
plt.figure()
plt.plot(f, 20 * np.log10(np.abs(h_fft)))
plt.grid()

# phase response
plt.figure()
plt.plot(f, np.unwrap(np.angle(h_fft)))
plt.grid()

# %% apply filter
ppg_filt = sig.lfilter(b, a, ppg)
plt.figure()
plt.plot(t, ppg)
plt.plot(t, ppg_filt)
plt.grid()

# %% detect <3 beats
thresh = np.max(ppg_filt) * 0.5
ppg_th = ppg_filt.copy()
ppg_th[ppg_th < thresh] = 0

plt.figure()
plt.plot(t, ppg_filt)
plt.plot([0, t[-1]], [thresh, thresh], "r")
plt.plot(t, ppg_th)
plt.grid()

# %% peak detection
peak_is = []
peak_amps = []
for i, amp in enumerate(ppg_th):
    if i == 0:
        continue
    if i == ppg_th.size - 1:
        break
    if ppg_th[i - 1] <= amp > ppg_th[i + 1]:
        peak_is.append(i)
        peak_amps.append(amp)

plt.figure()
plt.plot(t, ppg_th)
for i, amp in zip(peak_is, peak_amps):
    plt.plot(i / fs, amp, "or")
plt.grid()

# %% max pooling
t_min = 0.400  # s
peak_ts = t[peak_is]
peak_final_amps = []
peak_final_ts = []
prev_peak_t = None
for peak_t, peak in zip(peak_ts, peak_amps):
    if (prev_peak_t is None) or (
            peak_t - prev_peak_t > t_min
            ):
        peak_final_ts.append(peak_t)
        peak_final_amps.append(peak)
        prev_peak_t = peak_t

plt.figure()
plt.plot(t, ppg_th)
plt.plot(peak_final_ts, peak_final_amps, "or")
plt.grid()

# %% bpm
delta_peak_ts = np.diff(peak_final_ts)
bpm = 60 / delta_peak_ts
t_bpm = peak_final_ts[1:]

plt.figure()
plt.plot(t_bpm, bpm)
plt.grid()

# %% moving average
t_win = 2  # s
pos = 0
bpm_mavg = []
t_bpm_mavg = []
t_bpm = np.array(t_bpm)
while pos <= t_bpm[-1] - t_win:
    stop = pos + t_win
    bpm_in_win = (t_bpm >= pos) & (t_bpm <= stop)
    bpm_avg = np.mean(bpm[bpm_in_win])
    bpm_mavg.append(bpm_avg)
    t_bpm_mavg.append(pos + t_win / 2)
    pos += t_win / 4

plt.figure()
plt.plot(t_bpm, bpm, "x-")
plt.plot(t_bpm_mavg, bpm_mavg, "o-")
plt.grid()

# %% R-R diagram
delta_peak_ts = np.diff(peak_final_ts)
r_r_prev = delta_peak_ts[:-1]
r_r_next = delta_peak_ts[1:]

plt.figure()
plt.plot([0.2, 1.2], [0.2, 1.2], "--r")
plt.plot(r_r_prev, r_r_next, "o")
plt.grid()
