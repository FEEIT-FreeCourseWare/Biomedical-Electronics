#!/usr/bin/env python3
"""
EEG signal processing.

Copyright 2017 - 2019 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, March 2019
"""
import numpy as np
import mne
import pickle

# %% eyes open
file_name = 'data/S001R01.edf'
data = mne.io.read_raw_edf(file_name, preload=True)
eegs = data.get_data()
info = data.info
fs = info['sfreq']
channels = data.ch_names
channels = [x.replace('.', '') for x in channels]
t = np.arange(eegs.shape[1])/fs
eeg_eyes_open = eegs[channels.index('Cz'), :]

# %% eyes closed
file_name = 'data/S001R02.edf'
data = mne.io.read_raw_edf(file_name, preload=True)
eegs = data.get_data()
eeg_eyes_closed = eegs[channels.index('Cz'), :]

# %% save as pickle
with open('data/eeg_sample.pkl', 'wb') as f:
    pickle.dump((fs, eeg_eyes_open, eeg_eyes_closed), f, -1)
