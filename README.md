# Biomedical Electronics
Materials for the course on Biomedical Electronics taught in the 8th semester at the undergraduate level at the [Faculty of Electrical Engineering and Information Technologies](http://feit.ukim.edu.mk), [Ss Cyril and Methodius University of Skopje](http://ukim.edu.mk/), Macedonia.

The course focuses on human physiology, electronic measurement devices and algorithms for processing of biomedical signals.
All the algorithms are implemented using Python 3.6, using Numpy, Scipy, and Matplotlib.
The whole installation procedure to get going is documented in the lecture materials.

Content
-------

The lecture materials are in the included PDF.
These are written in Macedonian, as the course is held in Macedonian.
All the code is in the `code` folder, and the biomedical signal samples are in `code/data/`.
Comments in the code are written in English.


License
-------
All the software is distributed with the GNU General Public License v.3, given in `code/LICENSE`.
The lecture materials are distributed under the [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license.
The following signal samples are taken from [BioSPPy - Biosignal Processing in Python](https://github.com/PIA-Group/BioSPPy): `emg.txt`, and `ecg.txt`.
EEG signal samples are taken from the [EEG Motor Movement/Imagery Dataset](http://www.physionet.org/pn4/eegmmidb/) available on [PhysioNet](https://physionet.org/) corresponding to Subject 1, tasks 3, 5, and 6: `S001R03.edf`, `S001R05.edf`, and `S001R06.edf`.
For convenience channel C3 from tasks 1 and 2 is also made available as a pickle `eeg_sample.pkl`, which is used in the introductory EEG excercise `code/vezba2_eeg.py`.

The dataset for Human Activity Detection is the one released by the [Wireless Sensor Data Mining (WISDM) Lab](http://www.cis.fordham.edu/wisdm/).
The raw accelerometer signals are used in `vezba3_har.py` that are not included because of size limitations.
The data can be downloaded and extracted using the following code:

```bash
wget http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
mkdir path_to_git/code/data/har_wisdm
tar -xzvf WISDM_ar_latest.tar.gz -C path_to_git/code/data/har_wisdm
```


Branislav Gerazov

Departement of Electronics

[Faculty of Electrical Engineering and Information Technologies](http://feit.ukim.edu.mk)

[Ss Cyril and Methodius University of Skopje](http://ukim.edu.mk/)
