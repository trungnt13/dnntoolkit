# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division
import os

import numpy as np

import theano
from theano import tensor

import dnntoolkit

# ======================================================================
# Const
# ======================================================================
save_path = os.path.join(os.path.expanduser('~'), 'tmp', 'tmp.dat')

# ====== simulate data ====== #
n = 10e3
X = np.tile(np.arange(0, n)[:, None], (1, 100)).astype(np.int32)
y = dnntoolkit.tensor.to_categorical(np.random.randint(0, 10, n), n_classes=10)

str_dat = str(np.random.rand(1000, 1000))

# ======================================================================
# Save data
# ======================================================================
data = dnntoolkit.dataset(save_path, mode='w')

data['X'].append(X)
data['X'].duplicate(2)

data['y'].append(y)
data['y'].duplicate(2)

data['str'] = str_dat

data.close()

# ======================================================================
# Load data
# ======================================================================
data = dnntoolkit.dataset(save_path, mode='r')

for x in data['X'].iter(512, shuffle=False):
    print(x)

# Iteration is identical with the same seed
for x1 in data['X'].iter(512, shuffle=True, seed=dnntoolkit.MAGIC_SEED):
    pass
print()
for x2 in data['X'].iter(512, shuffle=True, seed=dnntoolkit.MAGIC_SEED):
    pass
print(x1 - x2)
