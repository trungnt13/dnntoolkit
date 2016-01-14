# python -m memory_profiler
# dataset used for benchmark: X=(None, 500, 120)
# dataset used for benchmark: y=(None, 2)

from __future__ import print_function, division

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
from theano import tensor

import numpy as np
import scipy as sp

import dnntoolkit
from memory_profiler import profile

import time
from itertools import izip
# ======================================================================
# Test
# ======================================================================
@profile
def main():
    t = dnntoolkit.dataset(
        '/volumes/backup/data/logmel_500_100_50_fre', 'r')
    # print(t)
    # print()

    X = t['X_train', 'X_valid']
    y = t['y_train', 'y_valid']
    it = izip(X.iter(512, shuffle=True, mode=1),
              y.iter(512, shuffle=True, mode=1))
    n_it = 0
    size = 0
    start = time.time()
    for i, j in it:
        if i.shape[0] != j.shape[0]:
            print('Shit happened')
        n_it += 1
        size += i.shape[0]
    end = time.time() - start
    print('Total %d iterations' % n_it)
    print('Time:', end)
    print('Avr/batch:', end / n_it)
    print('Avr/sample:', end / size)
    t.close()
    pass

if __name__ == '__main__':
    main()
# ======================================================================
# batch=512, block=20
# ======================================================================
# Total 27 iterations
# Time: 49.5420119762
# Avr/batch: 1.83488933245
# Avr/sample: 0.00398536014611

# Total 27 iterations
# Time: 43.8584640026
# Avr/batch: 1.62438755565
# Avr/sample: 0.00352815252213

# Total 27 iterations
# Time: 49.1226298809
# Avr/batch: 1.81935666226
# Avr/sample: 0.00395162335137

# Line #    Mem usage    Increment   Line Contents
# ================================================
#     19     92.2 MiB      0.0 MiB   @profile
#     20                             def main():
#     21     92.2 MiB      0.0 MiB       t = dnntoolkit.dataset(
#     22     93.3 MiB      1.1 MiB           '/volumes/backup/data/logmel_500_100_50_fre', 'r')
#     23     93.4 MiB      0.0 MiB       print(t)
#     24
#     25     93.4 MiB      0.1 MiB       X = t['X_train', 'X_valid']
#     26     93.4 MiB      0.0 MiB       y = t['y_train', 'y_valid']
#     27     93.4 MiB      0.0 MiB       it = izip(X.iter(512, block_size=20, shuffle=True, mode=0),
#     28     93.4 MiB      0.0 MiB                 y.iter(512, block_size=20, shuffle=True, mode=0))
#     29     93.4 MiB      0.0 MiB       n_it = 0
#     30     93.4 MiB      0.0 MiB       size = 0
#     31     93.4 MiB      0.0 MiB       start = time.time()
#     32   1678.8 MiB   1585.4 MiB       for i, j in it:
#     33   1678.8 MiB      0.0 MiB           if i.shape[0] != j.shape[0]:
#     34                                         print('Shit happened')
#     35   1678.8 MiB      0.0 MiB           n_it += 1
#     36   1678.8 MiB      0.0 MiB           size += i.shape[0]
#     37    305.6 MiB  -1373.2 MiB       end = time.time() - start
#     38    305.6 MiB      0.0 MiB       print('Total %d iterations' % n_it)
#     39    305.6 MiB      0.0 MiB       print('Time:', end)
#     40    305.6 MiB      0.0 MiB       print('Avr/batch:', end / n_it)
#     41    305.6 MiB      0.0 MiB       print('Avr/sample:', end / size)
#     42    307.4 MiB      1.7 MiB       t.close()
#     43    307.4 MiB      0.0 MiB       pass

# ======================================================================
# batch=512, block=10
# ======================================================================
# Total 27 iterations
# Time: 35.2815492153
# Avr/batch: 1.30672404501
# Avr/sample: 0.00283819075017

# Total 27 iterations
# Time: 41.0986790657
# Avr/batch: 1.52217329873
# Avr/sample: 0.00330614424147

# Total 27 iterations
# Time: 44.3804409504
# Avr/batch: 1.6437200352
# Avr/sample: 0.00357014246242

# Line #    Mem usage    Increment   Line Contents
# ================================================
#     20     92.1 MiB      0.0 MiB   @profile
#     21                             def main():
#     22     92.1 MiB      0.0 MiB       t = dnntoolkit.dataset(
#     23     93.2 MiB      1.1 MiB           '/volumes/backup/data/logmel_500_100_50_fre', 'r')
#     24                                 # print(t)
#     25                                 # print()
#     26
#     27     93.3 MiB      0.1 MiB       X = t['X_train', 'X_valid']
#     28     93.3 MiB      0.0 MiB       y = t['y_train', 'y_valid']
#     29     93.3 MiB      0.0 MiB       it = izip(X.iter(512, block_size=10, shuffle=True, mode=0),
#     30     93.3 MiB      0.0 MiB                 y.iter(512, block_size=10, shuffle=True, mode=0))
#     31     93.3 MiB      0.0 MiB       n_it = 0
#     32     93.3 MiB      0.0 MiB       size = 0
#     33     93.3 MiB      0.0 MiB       start = time.time()
#     34   1332.7 MiB   1239.4 MiB       for i, j in it:
#     35   1332.7 MiB      0.0 MiB           if i.shape[0] != j.shape[0]:
#     36                                         print('Shit happened')
#     37   1332.7 MiB      0.0 MiB           n_it += 1
#     38   1332.7 MiB      0.0 MiB           size += i.shape[0]
#     39    491.1 MiB   -841.6 MiB       end = time.time() - start
#     40    491.1 MiB      0.0 MiB       print('Total %d iterations' % n_it)
#     41    491.2 MiB      0.0 MiB       print('Time:', end)
#     42    491.2 MiB      0.0 MiB       print('Avr/batch:', end / n_it)
#     43    491.2 MiB      0.0 MiB       print('Avr/sample:', end / size)
#     44    493.5 MiB      2.4 MiB       t.close()
#     45    493.5 MiB      0.0 MiB       pass

