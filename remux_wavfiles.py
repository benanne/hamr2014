import sys
import os
import glob
import random

import h5py


SOURCE_PATH = "/home/sedielem/data/urbansound8k/UrbanSound8K/audio"
TARGET_PATH = "/home/sedielem/data/urbansound8k/audio_remuxed"

SIZE = 8732
N_FOLDS = 10


i = 0
for f in xrange(1, N_FOLDS + 1):
    print "fold %d" % f
    fold_path = os.path.join(SOURCE_PATH, "fold%d" % f)
    fold_target_path = os.path.join(TARGET_PATH, "fold%d" % f)

    if not os.path.exists(fold_target_path):
        os.makedirs(fold_target_path)

    clip_paths = glob.glob(os.path.join(fold_path, "*.wav"))

    for clip_path in clip_paths:
        print "  clip %d of %d" % (i + 1, SIZE)
        print "  %s" % clip_path

        clip_target_path = os.path.join(fold_target_path, os.path.basename(clip_path))
        command = "avconv -loglevel quiet -y -i %s %s" % (clip_path, clip_target_path)
        print command
        os.system(command)

        i += 1

print "done"
