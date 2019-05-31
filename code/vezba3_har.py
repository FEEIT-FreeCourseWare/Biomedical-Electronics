#!/usr/bin/env python3
"""
Human activity recognition.

The analysis is based on the WISDM HAR dataset that can be found here:
http://www.cis.fordham.edu/wisdm/dataset.php

Download the data and extract it in data/had_wisdm/

Copyright 2019 by Branislav Gerazov

See the file LICENSE for the license associated with this software.

Author(s):
  Branislav Gerazov, May 2019
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import neural_network
from sklearn import metrics
import bme

# %% load data
file_name = 'data/har_wisdm/WISDM_ar_v1.1_raw.txt'
columns = 'id act time x y z'.split()
data = pd.read_csv(file_name, header=None, names=columns, lineterminator=';')
fs = 20
data = data.dropna()

# %% data exploration - plot activities
activities = data.act.unique().tolist()
plt.figure(figsize=(7, 10))
for i, activity in enumerate(activities):
    mask_act = data.act == activity
    y = data.loc[mask_act, 'y'].values
    plt.plot(y[:200] + i*20, label=activity)

plt.grid()
plt.legend()
plt.tight_layout()

# %% structure data
data.id = data.id.astype(int)
subjects = data.id.unique().tolist()
train_subjects = subjects[:-7]
test_subjects = subjects[-7:]
x_train = None
y_train = None
x_test = None
y_test = None
for i_act, act in enumerate(activities):
    mask_act = data.act == act
    for sub in train_subjects:
        mask_sub = data.id == sub
        mask = mask_act & mask_sub
        if sum(mask) == 0:
            print(f'No {act} data for sub {sub}')
            continue
        # assert ~np.allclose(x, data.loc[mask, 'y'].values)
        x = data.loc[mask, 'y'].values
        __, __, x_feat = bme.get_spectrogram(fs, x, n_win=200)
        x_feat = x_feat.T
        mask_thresh = np.all(x_feat > -100, axis=1)
        x_feat = x_feat[mask_thresh, :]
        # y is one hot encoded
        y = np.zeros((x_feat.shape[0], 6))
        y[:, i_act] = 1
        if x_train is None:
            x_train = x_feat
            y_train = y
        else:
            x_train = np.concatenate((x_train, x_feat), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    for sub in test_subjects:
        mask_sub = data.id == sub
        mask = mask_act & mask_sub
        if sum(mask) == 0:
            print(f'No {act} data for sub {sub}')
            continue
        x = data.loc[mask, 'y'].values
        __, __, x_feat = bme.get_spectrogram(fs, x, n_win=200)
        x_feat = x_feat.T
        mask_thresh = np.all(x_feat > -100, axis=1)
        x_feat = x_feat[mask_thresh, :]
        # y is one hot encoded
        y = np.zeros((x_feat.shape[0], 6))
        y[:, i_act] = 1
        if x_test is None:
            x_test = x_feat
            y_test = y
        else:
            x_test = np.concatenate((x_test, x_feat), axis=0)
            y_test = np.concatenate((y_test, y), axis=0)

# %% plot features
x_axis = np.arange(x_train.shape[0])
y_axis = np.arange(x_train.shape[1])
bme.show_spectrogram(x_axis, y_axis, x_train.T)
plt.plot(y_train*80, lw=2)

x_axis = np.arange(x_test.shape[0])
y_axis = np.arange(x_test.shape[1])
bme.show_spectrogram(x_axis, y_axis, x_test.T)
plt.plot(y_test*80, lw=2)

# %% train
mlp = neural_network.MLPClassifier(
        hidden_layer_sizes=(100, 20, 10),
        alpha=1e-4,
        learning_rate_init=0.01,
        max_iter=3000,
        tol=1e-9,
        n_iter_no_change=20,
        early_stopping=False,
        activation='relu',
        verbose=1)
mlp.fit(x_train, y_train)

# %% test accuracy
y_pred = mlp.predict(x_test)
accuracy = mlp.score(x_test, y_test)
print(accuracy)


# %% calculate and plot confusion matrix
# adapted from
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
y_pred_prob = mlp.predict_proba(x_test)
cm = metrics.confusion_matrix(
        y_test.argmax(axis=1),
        y_pred_prob.argmax(axis=1))
print(cm)

# normalise and plot heat map
cm = cm / cm.sum(axis=0)
fig, ax = plt.subplots()
im = ax.imshow(cm, aspect='auto', interpolation='nearest', vmax=1, vmin=0)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=activities,
       yticklabels=activities,
       ylabel='Ground truth',
       xlabel='Prediction')
fig.colorbar(im)
fig.tight_layout()
