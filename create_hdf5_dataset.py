import sys
import os
import glob
import random
import re

import numpy as np
import librosa

import h5py


SOURCE_PATH = "/home/sedielem/data/urbansound8k/audio_remuxed"
TARGET_PATH = "/home/sedielem/data/urbansound8k/spectrograms_uncompressed_32.h5"
SIZE = 8732
N_FOLDS = 10

SAMPLERATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 32 # 128
# C = 10000


n_samples = SAMPLERATE * 4 # each clip is padded to 4 seconds (the maximal length)
spectrogram_length = n_samples / HOP_LENGTH + 1


d = h5py.File(TARGET_PATH, 'w')
d.create_dataset("spectrograms", (SIZE, N_MELS, spectrogram_length), dtype='float32')
d.create_dataset("folds", (SIZE,), dtype='int32')
d.create_dataset("fsids", (SIZE,), dtype='int32')
d.create_dataset("classids", (SIZE,), dtype='int32')
d.create_dataset("occurrenceids", (SIZE,), dtype='int32')
d.create_dataset("sliceids", (SIZE,), dtype='int32')

i = 0
for f in xrange(1, N_FOLDS + 1):
    print "fold %d" % f
    fold_path = os.path.join(SOURCE_PATH, "fold%d" % f)
    clip_paths = glob.glob(os.path.join(fold_path, "*.wav"))
    random.shuffle(clip_paths) # shuffle per fold in case we want to make minibatches by slicing.

    for clip_path in clip_paths:
        print "  clip %d of %d" % (i + 1, SIZE)
        print "  %s" % clip_path
        # extract spectrogram
        y, sr = librosa.core.load(clip_path, sr=SAMPLERATE, mono=True)
        y = y[:n_samples] # truncate to max length of 4 seconds if necessary
        
        y_padded = np.zeros((n_samples,), dtype='float32')
        offset = (n_samples - y.shape[0]) // 2
        y_padded[offset:offset + y.shape[0]] = y

        s = librosa.feature.melspectrogram(y_padded, sr=SAMPLERATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        # s = np.log(1 + C*s) # s = librosa.logamplitude(s)

        # parse filename
        # [fsID]-[classID]-[occurrenceID]-[sliceID].wav
        m = re.match("(\d+)-(\d+)-(\d+)-(\d+)\.wav", os.path.basename(clip_path))
        fsid = int(m.group(1))
        classid = int(m.group(2))
        occurrenceid = int(m.group(3))
        sliceid = int(m.group(4))

        # store data
        d['spectrograms'][i] = s
        d['folds'][i] = f
        d['fsids'][i] = fsid
        d['classids'][i] = classid
        d['occurrenceids'][i] = occurrenceid
        d['sliceids'][i] = sliceid

        i += 1

d.close()

print "done"
print "stored in %s" % TARGET_PATH