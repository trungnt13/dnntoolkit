from __future__ import print_function, division

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
from theano import tensor

import numpy as np
import scipy as sp

import lasagne
import dnntoolkit
import cPickle
import time
import h5py

from itertools import izip
from collections import OrderedDict

# ======================================================================
# Header
# ======================================================================
f = h5py.File('tmp1.hdf', 'w')
f['X1'] = np.zeros((43, 1)) + 1
f['Y1'] = np.zeros((43, 1)) - 1
f['X2'] = np.zeros((10, 1)) + 2
f['a'] = '111'
f['c'] = np.array([['a', 'b'], ['c', 'd']])
f.close()
f = h5py.File('tmp2.hdf', 'w')
f['X1'] = np.zeros((67, 1)) + 4
f['Y1'] = np.zeros((67, 1)) - 4
f['X3'] = np.zeros((130, 1)) + 3
f['Y3'] = np.zeros((130, 1)) - 3
f['Y'] = np.concatenate((np.zeros((43, 1)) - 1, np.zeros((130, 1)) - 3), axis=0)
f['X4'] = np.zeros((10, 1)) + 4
f['a'] = '222'
f['b'] = 'bbb'
f.close()

# ======================================================================
# Dataset
# ======================================================================
f1 = h5py.File('tmp1.hdf', 'r')
f2 = h5py.File('tmp2.hdf', 'r')
X = dnntoolkit._batch(['X1', 'X3'], [f1, f2])
y = dnntoolkit._batch(['Y1', 'Y3'], [f1, f2])
y1 = dnntoolkit._batch(['Y'], [f2])

start = 0.0
end = 1.0
shuffle = True
mode = 2
seed = 13

X_ = []
t = time.time()
for i in X.iter(batch_size=9, start=start, end=end,
       shuffle=shuffle, seed=seed, normalizer=None, mode=mode):
    X_.append(i)
print('Iter:', time.time() - t)
X_ = np.concatenate(X_, 0)
print(X_.shape)
print(np.sum(X_ == 1))
print(np.sum(X_ == 3))

y_ = []
t = time.time()
for i in y.iter(batch_size=9, start=start, end=end,
       shuffle=shuffle, seed=seed, normalizer=None, mode=mode):
    y_.append(i)
print('Iter:', time.time() - t)
y_ = np.concatenate(y_, 0)
print(y_.shape)
print(np.sum(y_ == -1))
print(np.sum(y_ == -3))

y1_ = []
t = time.time()
for i in y1.iter(batch_size=9, start=start, end=end,
       shuffle=shuffle, seed=seed, normalizer=None, mode=mode):
    y1_.append(i)
print('Iter:', time.time() - t)
y1_ = np.concatenate(y1_, 0)
print(y1_.shape)
print(np.sum(y1_ == -1))
print(np.sum(y1_ == -3))

# chekc if order is preserved
print(np.concatenate((X_, y_), axis=1).sum(axis=1))
try:
    print(np.concatenate((X_, y1_), axis=1).sum(axis=1))
except:
    print('X_ and y1_ have different size')

# print(b)
# print(b.value)
# print(b.shape)
# print(len(b))
# print(b[None, :].shape)
f1.close()
f2.close()
# ======================================================================
# End test
# ======================================================================
