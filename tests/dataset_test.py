# ======================================================================
# Author: TrungNT
# Run batch_test first
# ======================================================================
from __future__ import print_function, division
import os

import numpy as np

import theano
from theano import tensor

import dnntoolkit

import time

from itertools import izip
# ======================================================================
# Run batch test first
# ======================================================================
# os.system('python batch_test.py')

# ======================================================================
# Single file
# ======================================================================
print()
print('***********************************************')
print('************ Single: Writing test! ************')
print('***********************************************')
ds = dnntoolkit.dataset('tmp.hdf', mode='w')
ds['X'].append(np.tile(np.arange(50)[:, None], (1, 3)))
ds['y'] = np.arange(50)[:, None]
print(ds)
ds.close()

print()
print('***********************************************')
print('************ Single: Reading test! ************')
print('***********************************************')
ds = dnntoolkit.dataset('tmp.hdf', mode='r')

result = []
X = ds['X'].iter(5, start=0.0, end=1.0, shuffle=True)
y = ds['y'].iter(5, start=0.0, end=1.0, shuffle=True)
for i, j in izip(X, y):
    result.append(np.sum(i - j))

X = ds['X'].iter(5, start=0.5, end=0.1, shuffle=True)
y = ds['y'].iter(5, start=0.5, end=0.1, shuffle=True)
for i, j in izip(X, y):
    result.append(np.sum(i - j))

X = ds['X'].iter(5, start=0.1, end=0.8, shuffle=False)
y = ds['y'].iter(5, start=0.1, end=0.8, shuffle=False)
for i, j in izip(X, y):
    result.append(np.sum(i - j))

print('All results:')
print(result)
ds.close()

# ======================================================================
# Multiple files
# ======================================================================
print()
print('***************************************')
print('************ Reading test! ************')
print('***************************************')
ds = dnntoolkit.dataset(['tmp1.hdf', 'tmp2.hdf'], mode='r')
print(ds)
print(ds['a'])
print(ds['X1'])

print(ds['X1', 'X3'])
print(ds['Y1', 'Y3'])
print(np.sum(ds['X1', 'X3'][:] + ds['Y1', 'Y3'][:]))

result = []
X = ds['X1', 'X3'].iter(3, shuffle = True, mode = 0)
y = ds['Y1', 'Y3'].iter(3, shuffle = True, mode = 0)
for i, j in izip(X, y):
    result.append(np.sum(i + j)) # perfect fit would print 0

X = ds['X1', 'X3'].iter(3, shuffle = True, mode = 1)
y = ds['Y1', 'Y3'].iter(3, shuffle = True, mode = 1)
for i, j in izip(X, y):
    result.append(np.sum(i + j)) # perfect fit would print 0

X = ds['X1', 'X3'].iter(3, shuffle = True, mode = 2)
y = ds['Y1', 'Y3'].iter(3, shuffle = True, mode = 2)
for i, j in izip(X, y):
    result.append(np.sum(i + j)) # perfect fit would print 0

X = ds['X1', 'X3'].iter(3, start=0.1, end=0.4, shuffle = True, mode = 0)
y = ds['Y1', 'Y3'].iter(3, start=0.1, end=0.4, shuffle = True, mode = 0)
for i, j in izip(X, y):
    result.append(np.sum(i + j)) # perfect fit would print 0

X = ds['X1', 'X3'].iter(3, start=0.1, end=0.4, shuffle = True, mode = 1)
y = ds['Y1', 'Y3'].iter(3, start=0.1, end=0.4, shuffle = True, mode = 1)
for i, j in izip(X, y):
    result.append(np.sum(i + j)) # perfect fit would print 0

X = ds['X1', 'X3'].iter(3, start=0.1, end=0.4, shuffle = True, mode = 2)
y = ds['Y1', 'Y3'].iter(3, start=0.1, end=0.4, shuffle = True, mode = 2)
for i, j in izip(X, y):
    result.append(np.sum(i + j)) # perfect fit would print 0

print('All result:')
print(result)
ds.close()

print()
print('***************************************')
print('************ Writing test! ************')
print('***************************************')
ds = dnntoolkit.dataset(['tmp3.hdf', 'tmp4.hdf'], mode='w')
print(ds)
ds['X4'] = np.zeros((10, 1)) + 4

ds.set_write('all')
ds['X1'] = np.zeros((10, 1)) + 1

ds.set_write('tmp3.hdf')
ds['X3'] = np.zeros((10, 1)) + 3

ds.set_write(slice(None, None))
ds['all'] = 'hello'

ds.set_write(1)
ds['X4+1'] = np.zeros((10, 1)) + 5

ds.set_write(['tmp3.hdf', 'tmp4.hdf'])
ds['X3+4'] = np.zeros((10, 1)) + 7

print(ds)
ds.close()
