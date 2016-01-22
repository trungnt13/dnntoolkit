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
import h5py
import unittest
# ======================================================================
# Run batch test first
# ======================================================================
def generateData():
    f = h5py.File('tmp/tmp1.h5', 'w')
    f['X1'] = np.zeros((100, 1)) + 1
    f['Y1'] = np.zeros((100, 1)) - 1

    f['X3'] = np.arange(100)[:, None]
    f['Y4'] = -np.arange(100, 200)[:, None]

    f['X5'] = np.arange(100).reshape(-1, 2)

    f['a'] = '111'
    f['c'] = np.array([['a', 'b'], ['c', 'd']])
    f.close()
    f = h5py.File('tmp/tmp2.h5', 'w')
    f['X1'] = np.zeros((100, 1)) + 1
    f['X2'] = np.zeros((200, 1)) + 2
    f['Y2'] = np.zeros((200, 1)) - 2

    f['Y3'] = -np.arange(100)[:, None]
    f['X4'] = np.arange(100, 200)[:, None]

    f['X5'] = np.arange(100, 200).reshape(-1, 2)

    f['Y'] = np.asarray([-1] * 100 + [-2] * 200)[:, None]
    f['a'] = '222'
    f['b'] = 'bbb'
    f.close()

def cleanUp():
    try:
        os.remove('tmp/tmp1.h5')
        os.remove('tmp/tmp2.h5')
    except:
        pass

class BatchTest(unittest.TestCase):

    def setUp(self):
        self.f1 = h5py.File('tmp/tmp1.h5', 'r')
        self.f2 = h5py.File('tmp/tmp2.h5', 'r')

        self.X = dnntoolkit.batch(['X1', 'X2'], [self.f1, self.f2]) # n_X1=200; n_X2=200
        self.y = dnntoolkit.batch(['Y1', 'Y2'], [self.f1, self.f2]) # n_Y1=200; n_X2=200
        self.y1 = dnntoolkit.batch(['Y'], [self.f2])

        self.X34 = dnntoolkit.batch(['X3', 'X4'], [self.f1, self.f2])
        self.y34 = dnntoolkit.batch(['Y3', 'Y4'], [self.f2, self.f1])

        self.X12 = dnntoolkit.batch(['X1', 'X2'], [self.f1, self.f2])
        self.Y = dnntoolkit.batch('Y', self.f2)

    def tearDown(self):
        self.f1.close()
        self.f2.close()

    def test_cross_iter_2_hdf(self):
        start, end = np.random.rand(1)[0] / 2, np.random.rand(1)[0] / 2 + 0.5
        seed = 13
        for shuffle in (True, False):
            for mode in (0, 1, 2):
                X_ = np.concatenate(list(self.X34.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                seed=seed, normalizer=None, mode=mode)), 0)
                y_ = np.concatenate(list(self.y34.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                seed=seed, normalizer=None, mode=mode)), 0)
                self.assertEqual(np.sum(X_ + y_), 0.,
                    'X and y has different order, shuffle=' + str(shuffle) + ', mode=' + str(mode))
        for shuffle in (True, False):
            for mode in (0, 1, 2):
                X_ = np.concatenate(list(self.X.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                seed=seed, normalizer=None, mode=mode)), 0)
                y_ = np.concatenate(list(self.y.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                seed=seed, normalizer=None, mode=mode)), 0)
                self.assertEqual(np.sum(X_ + y_), 0.,
                    'X and y has different order, shuffle=' + str(shuffle) + ', mode=' + str(mode))

        start, end = 0., 1. # this one only support full dataset
        for shuffle in (False, ):
            for mode in (0, 2):
                X_ = np.concatenate(list(self.X12.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                seed=seed, normalizer=None, mode=mode)), 0)
                y_ = np.concatenate(list(self.Y.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                seed=seed, normalizer=None, mode=mode)), 0)
                self.assertEqual(np.sum(X_ + y_), 0.,
                    'X and y has different order, shuffle=' + str(shuffle) + ', mode=' + str(mode))

    def test_indexing(self):
        self.assertEqual(self.X34[:].ravel().tolist(), range(200))
        self.assertEqual(self.y34[:].ravel().tolist(), [-i for i in range(200)])

    def test_arithmetic(self):
        X5 = dnntoolkit.batch(['X5', 'X5'], [self.f1, self.f2])
        X = np.concatenate(
            (np.arange(100).reshape(-1, 2), np.arange(100, 200).reshape(-1, 2)), 0)
        self.assertEqual(X5.sum(0).tolist(), X.sum(0).tolist())
        self.assertEqual(X5.sum2(0).tolist(), np.power(X, 2).sum(0).tolist())
        self.assertEqual(X5.mean(0).tolist(), X.mean(0).tolist())
        self.assertEqual(X5.var(0).tolist(), X.var(0).tolist())

        self.assertEqual(X5.sum(1).tolist(), X.sum(1).tolist())
        self.assertEqual(X5.sum2(1).tolist(), np.power(X, 2).sum(1).tolist())
        self.assertEqual(X5.mean(1).tolist(), X.mean(1).tolist())
        self.assertEqual(X5.var(1).tolist(), X.var(1).tolist())

    def test_double_iteration(self):
        X12 = dnntoolkit.batch(['X1', 'X2'], [self.f1, self.f2])
        Y12 = dnntoolkit.batch('Y', self.f2)
        x = np.concatenate(list(X12.iter(9, start=0, end=1., shuffle=False, mode=0)), 0).ravel()
        y = np.concatenate(list(Y12.iter(9, start=0, end=1., shuffle=False, mode=0)), 0).ravel()
        self.assertEqual(np.sum(x + y), 0)
        x = np.concatenate(list(X12.iter(9, start=0, end=1., shuffle=True, mode=0)), 0).ravel()
        y = np.concatenate(list(Y12.iter(9, start=0, end=1., shuffle=True, mode=0)), 0).ravel()
        self.assertEqual(np.sum(x + y), 0)
        x = np.concatenate(list(X12.iter(9, start=0, end=1., shuffle=True, mode=2)), 0).ravel()
        y = np.concatenate(list(Y12.iter(9, start=0, end=1., shuffle=True, mode=2)), 0).ravel()
        self.assertEqual(np.sum(x + y), 0)
        x = np.concatenate(list(X12.iter(9, start=0, end=1., shuffle=True, mode=0)), 0).ravel()
        y = np.concatenate(list(Y12.iter(9, start=0, end=1., shuffle=True, mode=0)), 0).ravel()
        self.assertEqual(np.sum(x + y), 0)

    def test_single_iteration(self):
        l = np.concatenate(list(self.X34.iter(batch_size=9, mode=0, shuffle=False)), 0)
        self.assertEqual(l.ravel().tolist(), range(200))

        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=0, shuffle=False, start=0., end=0.5)), 0)
        self.assertEqual(l.ravel().tolist(), range(50) + range(100, 150))

        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=0, shuffle=False, start=0.5, end=0.8)), 0)
        self.assertEqual(l.ravel().tolist(), range(50, 80) + range(150, 180))

        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=0, shuffle=True, start=0.8, end=0.5)), 0)
        self.assertEqual(sorted(l.ravel().tolist()), range(50, 80) + range(150, 180))

        # ====== Mode=2 ====== #
        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=2, shuffle=True, start=0.5, end=0.8)), 0)
        self.assertEqual(sorted(l.ravel().tolist()), range(50, 80) + range(150, 180))

        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=2, shuffle=False, start=0.5, end=0.8)), 0)
        self.assertEqual(sorted(l.ravel().tolist()), range(50, 80) + range(150, 180))

class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.ds = dnntoolkit.dataset(['tmp/tmp1.h5', 'tmp/tmp2.h5'], mode='r')

    def tearDown(self):
        self.ds.close()

    def test_print(self):
        print()
        print(self.ds)

    def test_indexing(self):
        b = self.ds['X1', 'X2']

        self.assertEqual(tuple(b._key), ('X1', 'X1', 'X2'))
        self.assertEqual(b._hdf[0].filename, 'tmp/tmp1.h5')
        self.assertEqual(b._hdf[1].filename, 'tmp/tmp2.h5')
        self.assertEqual(b._hdf[2].filename, 'tmp/tmp2.h5')

        self.assertEqual(tuple(self.ds['a']), ('111', '222'))
# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    try:
        generateData()
        unittest.main()
    except:
        pass
    finally:
        cleanUp()
