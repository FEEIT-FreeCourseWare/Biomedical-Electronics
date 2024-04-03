#!/usr/bin/env python3
"""
Multichannel EEG processing.

Copyright 2017 - 2019 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, March 2017
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
import mne
import bme

# %% load eeg signals
data = mne.io.read_raw_edf('data/S001R05.edf', preload=True)
eegs = data.get_data()
info = data.info
ch_names = data.ch_names
fs = info['sfreq']
ch_names = [x.replace('.', '') for x in ch_names]

# %% select electrodes
eeg_cz = eegs[ch_names.index('Cz'), :]
eeg_c3 = eegs[ch_names.index('C3'), :]
t = np.arange(eeg_cz.size) / fs
anns = eegs[-1, :]
anns[anns < 1] = 1
anns -= 1  # vo opseg 0 - 2

# %% plot t
fig, ax = plt.subplots()
ax.plot(t, eeg_cz)
ax.plot(t, eeg_c3)
ax2 = ax.twinx()
ax2.plot(t, anns, 'g', lw=3, alpha=.7)
ax.grid()
ax.legend(['Cz', 'C3'])
fig.tight_layout()

# %% spectrogram
t_frame, f_frame, eeg_cz_spec = bme.get_spectrogram(fs, eeg_cz, n_win=80)
t_frame, f_frame, eeg_c3_spec = bme.get_spectrogram(fs, eeg_c3, n_win=80)

bme.show_spectrogram(t_frame, f_frame, eeg_cz_spec)
plt.plot(t, anns/2*75, 'r', lw=3, alpha=.7)
bme.show_spectrogram(t_frame, f_frame, eeg_c3_spec)
plt.plot(t, anns/2*75, 'r', lw=3, alpha=.7)

# %% BP filter 7 - 30 Hz
f_l = 7
order = 80
b = sig.firwin(order+1,
               f_l,
               pass_zero=False,
               fs=fs,
               )
f, h = sig.freqz(b, 1, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
eeg_cz_hp = sig.filtfilt(b, 1, eeg_cz)
eeg_c3_hp = sig.filtfilt(b, 1, eeg_c3)

# %% lp filter
f_h = 30
order = 50
b = sig.firwin(order+1,
               f_h,
               pass_zero=True,
               fs=fs,
               )
f, h = sig.freqz(b, 1, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
eeg_cz_filt = sig.filtfilt(b, 1, eeg_cz_hp)
eeg_c3_filt = sig.filtfilt(b, 1, eeg_c3_hp)

# %% filters for the EEG bands
order = 150
b_alpha = sig.firwin(order+1,
                     [8, 13],
                     pass_zero=False,
                     fs=fs,
                     )
b_beta = sig.firwin(order+1,
                    [13, 30],
                    pass_zero=False,
                    fs=fs,
                    )

# %% plot transfer
plt.figure()
f, h = sig.freqz(b_alpha, 1, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
f, h = sig.freqz(b_beta, 1, fs=fs)
plt.plot(f, 20*np.log10(np.abs(h)))
plt.grid()
plt.axis([0, 50, -80, 10])

# %% filter
eeg_cz_alpha = sig.filtfilt(b_alpha, 1, eeg_cz_filt)
eeg_c3_alpha = sig.filtfilt(b_alpha, 1, eeg_c3_filt)

eeg_cz_beta = sig.filtfilt(b_beta, 1, eeg_cz_filt)
eeg_c3_beta = sig.filtfilt(b_beta, 1, eeg_c3_filt)

# %% get energy
t_eng, eeg_cz_alpha_eng = bme.get_energy(fs, eeg_cz_alpha, n_win=80)
t_eng, eeg_cz_beta_eng = bme.get_energy(fs, eeg_cz_beta, n_win=80)

t_eng, eeg_c3_alpha_eng = bme.get_energy(fs, eeg_c3_alpha, n_win=80)
t_eng, eeg_c3_beta_eng = bme.get_energy(fs, eeg_c3_beta, n_win=80)

# %% plot
fig, ax = plt.subplots()
ax.plot(t_eng, eeg_cz_alpha_eng)
ax.plot(t_eng, eeg_c3_alpha_eng)
ax2 = ax.twinx()
ax2.plot(t, anns, 'g', lw=3, alpha=.7)
ax.legend(['Cz', 'C3'])
ax.grid()
fig.tight_layout()
