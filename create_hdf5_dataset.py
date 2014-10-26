import sys
import os
import glob
import random
import re

import numpy as np
import librosa

import h5py


SOURCE_PATH = "/home/sedielem/data/urbansound8k/UrbanSound8K/audio"
TARGET_PATH = "/home/sedielem/data/urbansound8k/spectrograms.h5"
SIZE = 8732
N_FOLDS = 10

SAMPLERATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
C = 10000


spectrogram_length = (SAMPLERATE * 4 - N_FFT) / HOP_LENGTH + 1 # each clip is 4 seconds

d = h5py.File(TARGET_PATH, 'w')
d.create_dataset("spectrograms", (SIZE, N_MELS, spectrogram_length), dtype='float32')
d.create_dataset("folds", (SIZE,), dtype='int32')
d.create_dataset("fsids", (SIZE,), dtype='int32')
d.create_dataset("classids", (SIZE,), dtype='int32')
d.create_dataset("occurrenceids", (SIZE,), dtype='int32')
d.create_dataset("sliceids", (SIZE,), dtype='int32')

i = 0
for f in xrange(NUM_FOLDS):
    print "fold %d" % f
    fold_path = os.path.join(SOURCE_PATH, "fold%d" % f)
    clip_paths = glob.glob(os.path.join(fold_path, "*.wav"))
    random.shuffle(clip_paths) # shuffle per fold in case we want to make minibatches by slicing.

    for clip_path in clip_paths:
        print "  clip %d of %d" % (i + 1, SIZE)
        # extract spectrogram
        y, sr = librosa.core.load(clip_path, sr=SAMPLERATE, mono=True)
        s = librosa.feature.melspectrogram(y, sr=SAMPLERATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        s = np.log(1 + C*s) # s = librosa.logamplitude(s)

        # parse filename
        # [fsID]-[classID]-[occurrenceID]-[sliceID].wav
        m = re.match("(\d+)-(\d)-(\d)-(\d+)\.wav", path)
        fsid = int(m.group(1))
        classid = int(m.group(2))
        occurrenceid = int(m.group(3))
        sliceid = int(m.group(4))

        # store data
        d['spectrograms'][i] = s
        d['folds'][i] = f
        d['fsids'][i] = fsid
        d['classid'][i] = classid
        d['occurrenceid'][i] = occurrenceid
        d['sliceid'][i] = sliceid

        i += 1

d.close()

print "done"
print "stored in %s" % TARGET_PATH