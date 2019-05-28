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

# %% plot transfer
plt.figure()
f, h = sig.freqz(b_delta, a_delta, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
f, h = sig.freqz(b_theta, a_theta, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
f, h = sig.freqz(b_alpha, a_alpha, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
f, h = sig.freqz(b_beta, a_beta, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
f, h = sig.freqz(b_gamma, a_gamma, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
plt.grid()
plt.axis([0, 50, -80, 10])

# %% filter
eeg_open_delta = sig.lfilter(b_delta, a_delta, eeg_open_filt)
eeg_closed_delta = sig.lfilter(b_delta, a_delta, eeg_closed_filt)

eeg_open_theta = sig.lfilter(b_theta, a_theta, eeg_open_filt)
eeg_closed_theta = sig.lfilter(b_theta, a_theta, eeg_closed_filt)

eeg_open_alpha = sig.lfilter(b_alpha, a_alpha, eeg_open_filt)
eeg_closed_alpha = sig.lfilter(b_alpha, a_alpha, eeg_closed_filt)

eeg_open_beta = sig.lfilter(b_beta, a_beta, eeg_open_filt)
eeg_closed_beta = sig.lfilter(b_beta, a_beta, eeg_closed_filt)

eeg_open_gamma = sig.lfilter(b_gamma, a_gamma, eeg_open_filt)
eeg_closed_gamma = sig.lfilter(b_gamma, a_gamma, eeg_closed_filt)

# %% plot open
plt.figure(figsize=(10, 10))
plt.subplot(511)
plt.plot(t, eeg_open_delta)
plt.axis([10, 20, -1e-4, 1e-4])
plt.subplot(512)
plt.plot(t, eeg_open_theta)
plt.axis([10, 20, -1e-4, 1e-4])
plt.subplot(513)
plt.plot(t, eeg_open_alpha)
plt.axis([10, 20, -1e-4, 1e-4])
plt.subplot(514)
plt.plot(t, eeg_open_beta)
plt.axis([10, 20, -1e-4, 1e-4])
plt.subplot(515)
plt.plot(t, eeg_open_gamma)
plt.axis([10, 20, -1e-4, 1e-4])
plt.tight_layout()

# %% plot closed
eegs_closed = [
        eeg_closed_delta,
        eeg_closed_theta,
        eeg_closed_alpha,
        eeg_closed_beta,
        eeg_closed_gamma,
        ]
plt.figure(figsize=(10, 10))
for i, eeg in enumerate(eegs_closed):
    plt.subplot(5, 1, i+1)
    plt.plot(t, eeg)
    plt.axis([10, 20, -1e-4, 1e-4])
plt.tight_layout()

# %% calculate power
eegs_open = [
        eeg_open_delta,
        eeg_open_theta,
        eeg_open_alpha,
        eeg_open_beta,
        eeg_open_gamma,
        ]
eegs_closed = np.array(eegs_closed)
eegs_open = np.array(eegs_open)

eegs_closed_pow = np.sum(eegs_closed**2, axis=1)
eegs_open_pow = np.sum(eegs_open**2, axis=1)

# %% plot bar
plt.figure()
plt.bar(np.arange(5)+0.2,  # bar position
        eegs_open_pow,
        width=0.4)
plt.bar(np.arange(5)+0.6,
        eegs_closed_pow,
        width=0.4)
plt.legend(['eyes open', 'eyes closed'])
plt.xticks(np.arange(5)+0.4, 'delta theta alpha beta gamma'.split())

# %% normalised power
eegs_closed_pow_norm = eegs_closed_pow / np.sum(eegs_closed_pow)
eegs_open_pow_norm = eegs_open_pow / np.sum(eegs_open_pow)

plt.figure()
plt.bar(np.arange(5)+0.2,
        eegs_open_pow_norm,
        width=0.4)
plt.bar(np.arange(5)+0.6,
        eegs_closed_pow_norm,
        width=0.4)
plt.legend(['eyes open', 'eyes closed'])
plt.xticks(np.arange(5)+0.4, 'delta theta alpha beta gamma'.split())
