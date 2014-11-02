import numpy as np 
import theano
import theano.tensor as T

import nntools as nn
from nntools.theano_extensions import conv

import h5py

from collections import OrderedDict

import skimage


# DATASET_PATH = "/home/sedielem/data/urbansound8k/spectrograms.h5"
DATASET_PATH = "data/spectrograms.h5" # "data/spectrograms_uncompressed.h5"
NUM_CLASSES = 10
CHUNK_SIZE = 8 * 4096
NUM_CHUNKS = 1000
NUM_TIMESTEPS_AUG = 110
NUM_FREQ_COMPONENTS_AUG = 128
MB_SIZE = 128
LEARNING_RATE = 0.01 # 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EVALUATE_EVERY = 1 # always validate since it's fast enough
# SOFTMAX_LAMBDA = 0.01
COMPRESSION_CONSTANT = 10000


d = h5py.File(DATASET_PATH, 'r')

folds = d['folds'][:]
idcs_eval = (folds == 9) | (folds == 10)
idcs_train = ~idcs_eval

spectrograms = d['spectrograms'][:]

data_train = spectrograms[idcs_train, :, :]
labels_train = d['classids'][idcs_train]

num_examples_train, num_freq_components, num_timesteps = data_train.shape
num_batches_train = CHUNK_SIZE // MB_SIZE

offset_eval = (num_timesteps - NUM_TIMESTEPS_AUG) // 2
data_eval = spectrograms[idcs_eval, :, :]
labels_eval = d['classids'][idcs_eval]

num_examples_eval = data_eval.shape[0]

def build_chunk(data, labels, chunk_size, num_timesteps_aug):
    chunk = np.empty((chunk_size, num_freq_components, num_timesteps_aug), dtype='float32')
    idcs = np.random.randint(0, data.shape[0], chunk_size)
    offsets = np.random.randint(0, num_timesteps - num_timesteps_aug, chunk_size)

    for l in xrange(chunk_size):
        chunk[l] = data[idcs[l], :, offsets[l]:offsets[l] + num_timesteps_aug]

    return chunk, labels[idcs]

# def build_chunk(data, labels, chunk_size, num_timesteps_aug, num_freq_components_aug):
#     chunk = np.empty((chunk_size, num_freq_components, num_timesteps_aug), dtype='float32')
#     idcs = np.random.randint(0, data.shape[0], chunk_size)
#     offsets_time = np.random.uniform(0, num_timesteps - num_timesteps_aug, chunk_size)
#     offsets_freq = np.random.uniform(0, num_freq_components - num_freq_components_aug, chunk_size)


#     for l in xrange(chunk_size):
#         # chunk[l] = data[idcs[l], :, offsets[l]:offsets[l] + num_timesteps_aug]
#         pass # TODO

#     chunk = np.log(1 + COMPRESSION_CONSTANT*chunk) # compression
#     # TODO: librosa?

#     return chunk, labels[idcs]



def train_chunks_gen(num_chunks, chunk_size, num_timesteps_aug): # , num_freq_components_aug):
    for k in xrange(num_chunks):
        yield build_chunk(data_train, labels_train, chunk_size, num_timesteps_aug) # , num_freq_components_aug)

train_gen = train_chunks_gen(NUM_CHUNKS, CHUNK_SIZE, NUM_TIMESTEPS_AUG) # , NUM_FREQ_COMPONENTS_AUG)

# generate fixed evaluation chunk
chunk_eval, chunk_eval_labels = build_chunk(data_eval, labels_eval, CHUNK_SIZE, NUM_TIMESTEPS_AUG) # , NUM_FREQ_COMPONENTS_AUG)
num_batches_eval = chunk_eval.shape[0] // MB_SIZE



## architecture
# 10 <=(3)= 12 <=[2]= 24 <=(3)= 26 <=[2]= 52 <=(3)= 54 <=[2]= 108 <=(3)= 110

l_in = nn.layers.InputLayer((MB_SIZE, num_freq_components, NUM_TIMESTEPS_AUG))

l1a = nn.layers.Conv1DLayer(l_in, num_filters=32, filter_length=3, convolution=conv.conv1d_md)
l1 = nn.layers.FeaturePoolLayer(l1a, ds=2, axis=2) # abusing the feature pool layer as a regular 1D max pooling layer

l2a = nn.layers.Conv1DLayer(l1, num_filters=32, filter_length=3, convolution=conv.conv1d_md)
l2b = nn.layers.NINLayer(l2a, num_units=32)
l2 = nn.layers.FeaturePoolLayer(l2b, ds=2, axis=2)

l3a = nn.layers.Conv1DLayer(l2, num_filters=32, filter_length=3, convolution=conv.conv1d_md)
l3b = nn.layers.NINLayer(l3a, num_units=32)
l3 = nn.layers.FeaturePoolLayer(l3b, ds=2, axis=2)

l4a = nn.layers.Conv1DLayer(l3, num_filters=32, filter_length=3, convolution=conv.conv1d_md)
l4 = nn.layers.GlobalPoolLayer(l4a) # global mean pooling across the time axis

l5 = nn.layers.DenseLayer(nn.layers.dropout(l4, p=0.5), num_units=64)

l6 = nn.layers.DenseLayer(nn.layers.dropout(l5, p=0.5), num_units=NUM_CLASSES, nonlinearity=nn.nonlinearities.identity) # , nonlinearity=T.nnet.softmax)

all_params = nn.layers.get_all_params(l6)
param_count = sum([np.prod(p.get_value().shape) for p in all_params])
print "parameter count: %d" % param_count

def multiclass_svm(y, t, l2=True): # t is one-hot
    y_correct = (y * t).sum(1).dimshuffle(0, 'x')
    d = T.maximum(0, 1 - (y_correct - y)) # the margin between the correct x and all others should be >= 1

    # average over examples (axis=0) and classes (axis=1)
    if l2:
        return T.mean(d**2) # L2 SVM loss
    else:
        return T.mean(d) # true hinge loss

# TODO: adapt

obj = nn.objectives.Objective(l6, loss_function=multiclass_svm)

loss_train = obj.get_loss()
loss_eval = obj.get_loss(deterministic=True)

updates_train = OrderedDict(nn.updates.nesterov_momentum(loss_train, all_params, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY))
# updates_train[l6.W] += SOFTMAX_LAMBDA * T.mean(T.sqr(l6.W)) # L2 loss on the softmax weights to avoid saturation

y_pred_train = T.argmax(l6.get_output(), axis=1)
y_pred_eval = T.argmax(l6.get_output(deterministic=True), axis=1)


## compile

X_train = nn.utils.shared_empty(dim=3)
y_train = nn.utils.shared_empty(dim=1)

X_eval = theano.shared(chunk_eval)
y_eval = theano.shared(chunk_eval_labels)


index = T.lscalar("index")

acc_train = T.mean(T.eq(y_pred_train, y_train[index * MB_SIZE:(index + 1) * MB_SIZE]))
acc_eval = T.mean(T.eq(y_pred_eval, y_eval[index * MB_SIZE:(index + 1) * MB_SIZE]))

givens_train = {
    l_in.input_var: X_train[index * MB_SIZE:(index + 1) * MB_SIZE],
    obj.target_var: nn.utils.one_hot(y_train[index * MB_SIZE:(index + 1) * MB_SIZE], NUM_CLASSES),
}
iter_train = theano.function([index], [loss_train, acc_train], givens=givens_train, updates=updates_train)

# # DEBUG
# from pylearn2.devtools.nan_guard import NanGuardMode
# mode = NanGuardMode(True, True, True)
# iter_train = theano.function([index], [loss_train, acc_train], givens=givens_train, updates=updates_train, mode=mode)

# debug_iter_train = theano.function([index], loss_train, givens=givens_train) # compute loss but don't compute updates

givens_eval = {
    l_in.input_var: X_eval[index * MB_SIZE:(index + 1) * MB_SIZE],
    obj.target_var: nn.utils.one_hot(y_eval[index * MB_SIZE:(index + 1) * MB_SIZE], NUM_CLASSES),
}
iter_eval = theano.function([index], [loss_eval, acc_eval], givens=givens_eval)

pred_train = theano.function([index], y_pred_train, givens=givens_train, on_unused_input='ignore')
pred_eval = theano.function([index], y_pred_eval, givens=givens_eval, on_unused_input='ignore')

## train

for k, (chunk_data, chunk_labels) in enumerate(train_gen):
    print "chunk %d (%d of %d)" % (k, k + 1, NUM_CHUNKS)

    print "  load data onto GPU"
    X_train.set_value(chunk_data)
    y_train.set_value(chunk_labels.astype(theano.config.floatX)) # can't store integers

    print "  train"
    losses_train = []
    accs_train = []
    for b in xrange(num_batches_train):
        # db_loss = debug_iter_train(b)
        # print "DEBUG DB_LOSS %.8f" % db_loss
        # if np.isnan(db_loss):
        #     raise RuntimeError("db_loss is NaN")

        loss_train, acc_train = iter_train(b)
        # print "DEBUG MIN INPUT %.8f" % chunk_data[b*MB_SIZE:(b+1)*MB_SIZE].min()
        # print "DEBUG MAX INPUT %.8f" % chunk_data[b*MB_SIZE:(b+1)*MB_SIZE].max()
        # print "DEBUG PARAM STD " + " ".join(["%.4f" % p.get_value().std() for p in all_params])
        # print "DEBUG LOSS_TRAIN %.8f" % loss_train # TODO DEBUG
        if np.isnan(loss_train):
            raise RuntimeError("loss_train is NaN")

        losses_train.append(loss_train)
        accs_train.append(acc_train)

    avg_loss_train = np.mean(losses_train)
    avg_acc_train = np.mean(accs_train)
    print "  avg training loss: %.5f" % avg_loss_train
    print "  avg training accuracy: %.3f%%" % (avg_acc_train * 100)

    if (k + 1) % EVALUATE_EVERY == 0:
        print "  evaluate"
        losses_eval = []
        accs_eval = []
        for b in xrange(num_batches_eval):
            loss_eval, acc_eval = iter_eval(b)
            if np.isnan(loss_eval):
                raise RuntimeError("loss_eval is NaN")

            losses_eval.append(loss_eval)
            accs_eval.append(acc_eval)

        avg_loss_eval = np.mean(losses_eval)
        avg_acc_eval = np.mean(accs_eval)
        print "  avg evaluation loss: %.5f" % avg_loss_eval
        print "  avg evaluation accuracy: %.3f%%" % (avg_acc_eval * 100)

