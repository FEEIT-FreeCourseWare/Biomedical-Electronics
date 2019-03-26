#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing EMG signal from BioSPPy.

Created on Tue Mar 26 14:15:31 2019

@author: dsp
"""
#%% imports
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack as fft

#%% init
file_name = 'emg.txt'
emg = np.loadtxt(file_name)
fs = 1000
n_bit = 12

#%% normalise
emg = emg - 2**(n_bit-1)
emg = emg / 2**(n_bit-1)
t = np.arange(0, emg.size/fs, 1/fs)

#%% plot
plt.figure()
plt.plot(t, emg)
plt.grid()

#%% spectrum
x = np.ceil(np.log2(emg.size))
n_fft = np.int(2**x)
emg_fft = fft.fft(emg, n_fft)
emg_amp = np.abs(emg_fft)
emg_pha = np.angle(emg_fft)
# overlap duplicates
n_keep = np.int(n_fft/2 + 1)
emg_amp = emg_amp[:n_keep]
emg_amp[1:-1] = 2 * emg_amp[1:-1]
emg_amp = emg_amp / emg.size
# log
emg_db = 20 * np.log10(emg_amp)

f = np.linspace(0, fs/2, n_keep)
#%% plot 
plt.figure()
plt.plot(f, emg_db)
plt.grid()

#%% spectrum func
import bme
f, emg_spec = bme.get_spectrum(fs, 
                               emg, 
                               n_fft)
#%% spectrogram
# init
n_win = 256  # 2**x
n_half = np.int(n_win / 2)
n_hop = n_half # Hann, Hamming
pad = np.zeros(n_half)
emg_pad = np.concatenate((pad,
                          emg,
                          pad,
                          ))
pos = 0
frames = None
while pos <= emg_pad.size - n_win:
    frame = emg_pad[pos : pos+n_win]
    f, frame_spec = bme.get_spectrum(fs,
                                     frame,
                                     n_win)
    frame_2d = frame_spec[:, np.newaxis]
    
    if frames is None:
        frames = frame_2d
    else:
        frames = np.concatenate((frames,
                                 frame_2d),
                                 axis=1,
                                 )
    pos += n_hop
#%%
import bme
n_win = 256
emg_nodc = emg - emg.mean()
t, f, emg_spectrogram = bme.get_spectrogram(fs, 
                                            emg_nodc, 
                                            n_win)
#%% plot spectrogram
plt.figure()
plt.imshow(emg_spectrogram,
           extent=[0, t[-1], 0, f[-1]],
           aspect='auto',
           origin='lower',
           vmin=-90,
           )
plt.colorbar()

#%% filter highpass 50 Hz 
from scipy import signal as sig
f_h = 100
order = 4
b, a = sig.iirfilter(order,
                     f_h/(fs/2),
                     btype='highpass',
                     ftype='butter')
f, h_f = sig.freqz(b, a, fs=fs)
plt.figure()
plt.subplot(2,1,1)
plt.plot(f, 20*np.log10(np.abs(h_f)))
plt.grid()
plt.subplot(2,1,2)
plt.plot(f, np.unwrap(np.angle(h_f)))
plt.grid()

#%% impulse response
excite = np.zeros(500)
excite[0] = 1  # Dirac
h_n = sig.lfilter(b, a, excite)
plt.figure()
plt.plot(h_n)
plt.grid()

#%% filter emg
emg_filt = sig.lfilter(b, a, emg_nodc)
t, f, emg_filt_specgram = bme.get_spectrogram(
        fs, emg_filt, n_win)
plt.figure()
plt.imshow(emg_filt_specgram,
           extent=[0, t[-1], 0, f[-1]],
           aspect='auto',
           origin='lower',
           vmin=-90,
           )
plt.colorbar()
#%% plot t
t = np.arange(0, emg.size/fs, 1/fs)
plt.figure()
plt.plot(t, emg)
plt.plot(t, emg_filt, lw=3, alpha=.5)
plt.grid()

#%% full wave rectification
emg_rect = np.abs(emg_filt)

# moving average window (lowpass)
t_win = .05  # 50 ms
n_win = np.int(t_win * fs)
win = sig.get_window('parzen', n_win)
win = win / np.sum(win)
# symmetric pad
n_pad = np.int(n_win / 2)
emg_pad = np.concatenate((
        emg_rect[n_pad:0:-1],
        emg_rect,
        emg_rect[-1:-n_pad:-1],
        ))
# filter
emg_mavg = sig.convolve(emg_pad, win,
                        'same')
emg_mavg = emg_mavg[n_pad:-n_pad+1]


#%% threshold
thresh = 1.2*emg_mavg.mean() \
         + 2*emg_mavg.std()

# plot
plt.figure()
plt.plot(t, emg_rect)
plt.plot(t, emg_mavg, alpha=.8)
plt.plot([t[0], t[-1]], [thresh, thresh],
         c='r',
         lw=4)
#plt.hlines(thresh, t[0], t[-1],
#           lw=4)
plt.grid()

#%% onset detection
over = np.where(emg_mavg > thresh)[0]
under = np.where(emg_mavg <= thresh)[0]
onsets = np.intersect1d(over-1, under)
onsets += 1

plt.figure()
plt.plot(t, emg_rect)
plt.plot(t, emg_mavg, alpha=.8)
plt.plot([t[0], t[-1]], [thresh, thresh],
         c='r',
         lw=4)
for onset in onsets:
    plt.plot([t[onset], t[onset]],
             [0, emg_rect.max()],
             c='c', lw=4)
plt.grid()

#%% moving average
t_win = 2  # 50 ms
n_win = np.int(t_win * fs)
win = sig.get_window('parzen', n_win)
win = win / np.sum(win)
onset_sig = np.zeros(emg_rect.size)
onset_sig[onsets] = 1
onset_mavg = sig.convolve(onset_sig,
                          win,
                          'same')
plt.figure()
plt.plot(t, emg_rect)
plt.plot(t, emg_mavg, alpha=.8)
plt.plot([t[0], t[-1]], [thresh, thresh],
         c='r',
         lw=3)
plt.plot(t, onset_mavg, c='c',
         lw=4, alpha=.8)

plt.grid()







