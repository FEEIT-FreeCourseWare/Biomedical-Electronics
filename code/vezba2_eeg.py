#!/usr/bin/env python3
"""
EEG signal processing.

Copyright 2017 - 2019 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, March 2017
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
import pickle
import bme

# %% load eeg signals
with open('data/eeg_sample.pkl', 'rb') as f:
    data = pickle.load(f)

fs, eeg_open, eeg_closed = data
t = np.arange(eeg_open.size) / fs

# %% plot t
plt.figure()
plt.subplot(211)
plt.plot(t, eeg_open)
plt.grid()
plt.subplot(212)
plt.plot(t, eeg_closed)
plt.grid()

# %% calculate spectrum
f, eeg_closed_amp = bme.get_spectrum(fs, eeg_closed)
f, eeg_open_amp = bme.get_spectrum(fs, eeg_open)

# %% plot spectrum
plt.figure()
plt.subplot(211)
plt.plot(f, eeg_open_amp)
plt.grid()
plt.subplot(212)
plt.plot(f, eeg_closed_amp)
plt.grid()

# %% spectrogram
t_frame, f_frame, eeg_open_spec = bme.get_spectrogram(
        fs, eeg_open, n_win=80)
t_frame, f_frame, eeg_closed_spec = bme.get_spectrogram(
        fs, eeg_closed, n_win=80)

bme.show_spectrogram(t_frame, f_frame, eeg_open_spec)
bme.show_spectrogram(t_frame, f_frame, eeg_closed_spec)

# %% BP filter 0.5 - 30 Hz
f_l = 0.5
order = 5
b, a = sig.iirfilter(order, f_l/(fs/2), btype='high', ftype='butter')
f, h = sig.freqz(b, a, fs=fs)
plt.figure()
plt.plot(f, 20*np.log10(np.abs(h)))
eeg_open_hp = sig.lfilter(b, a, eeg_open)
eeg_closed_hp = sig.lfilter(b, a, eeg_closed)

# %% lp filter
f_h = 45
order = 9
b, a = sig.iirfilter(order, f_h/(fs/2), btype='low', ftype='butter')
f, h = sig.freqz(b, a, fs=fs)
plt.figure()
plt.plot(f, 20*np.log10(np.abs(h)))
eeg_open_filt = sig.lfilter(b, a, eeg_open_hp)
eeg_closed_filt = sig.lfilter(b, a, eeg_closed_hp)

# %% filters for the EEG bands
b_delta, a_delta = sig.iirfilter(5, 4, btype='low', fs=fs)
b_theta, a_theta = sig.iirfilter(5, [4, 8], btype='band', fs=fs)
b_alpha, a_alpha = sig.iirfilter(5, [8, 13], btype='band', fs=fs)
b_beta, a_beta = sig.iirfilter(5, [13, 30], btype='band', fs=fs)
b_gamma, a_gamma = sig.iirfilter(5, 30, btype='high', fs=fs)

filter_bank = [
        (b_delta, a_delta),
        (b_theta, a_theta),
        (b_alpha, a_alpha),
        (b_beta, a_beta),
        (b_gamma, a_gamma),
        ]
bands = 'delta theta alpha beta gamma'.split()

# %% plot transfer
plt.figure()
for (b, a), band in zip(filter_bank, bands):
    f, h = sig.freqz(b, a, fs=fs)
    plt.plot(f, 20*np.log10(np.abs(h)), lw=3, alpha=.7, label=band)
plt.grid()
plt.legend()
plt.axis([0, f[-1], -60, 5])
plt.tight_layout()

# %% filter
eeg_open_bands = []
eeg_closed_bands = []
for b, a in filter_bank:
    eeg_open_bands.append(sig.lfilter(b, a, eeg_open_filt))
    eeg_closed_bands.append(sig.lfilter(b, a, eeg_closed_filt))

# %% plot bands open
plt.figure(figsize=(10, 10))
for i, eeg in enumerate(eeg_open_bands):
    plt.subplot(5, 1, i+1)
    plt.plot(t, eeg)
    plt.axis([10, 20, -1e-4, 1e-4])
plt.tight_layout()

# %% plot bands closed
plt.figure(figsize=(10, 10))
for i, eeg in enumerate(eeg_closed_bands):
    plt.subplot(5, 1, i+1)
    plt.plot(t, eeg)
    plt.axis([10, 20, -1e-4, 1e-4])
plt.tight_layout()

# %% calculate power
eeg_open_bands = np.array(eeg_open_bands)
eeg_closed_bands = np.array(eeg_closed_bands)

eegs_open_pow = np.mean(eeg_open_bands**2, axis=1)
eegs_closed_pow = np.mean(eeg_closed_bands**2, axis=1)

# %% plot bar
plt.figure()
plt.bar(np.arange(5)+0.2, eegs_open_pow, width=0.4)
plt.bar(np.arange(5)+0.6, eegs_closed_pow, width=0.4)
plt.legend(['eyes open', 'eyes closed'])
plt.xticks(np.arange(5)+0.4, bands)

# %% relative power
eegs_closed_pow_norm = eegs_closed_pow / np.sum(eegs_closed_pow)
eegs_open_pow_norm = eegs_open_pow / np.sum(eegs_open_pow)

# %% plot norm pow bar
plt.figure()
plt.bar(np.arange(5)+0.2, eegs_open_pow_norm, width=0.4)
plt.bar(np.arange(5)+0.6, eegs_closed_pow_norm, width=0.4)
plt.legend(['eyes open', 'eyes closed'])
plt.xticks(np.arange(5)+0.4, bands)
plt.tight_layout()
