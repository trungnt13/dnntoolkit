# ======================================================================
# Author: TrungNT
# Run batch_test first
# ======================================================================
from __future__ import print_function, division
import os

import numpy as np
import scipy as sp

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

    f['X3'] = np.arange(100).reshape(-1, 2)
    f['Y4'] = -np.arange(100, 200).reshape(-1, 2)

    f['X5'] = np.arange(100).reshape(-1, 2)

    f['a'] = '111'
    f['c'] = np.array([['a', 'b'], ['c', 'd']])
    f.close()
    f = h5py.File('tmp/tmp2.h5', 'w')
    f['X1'] = np.zeros((100, 1)) + 1
    f['X2'] = np.zeros((200, 1)) + 2
    f['Y2'] = np.zeros((200, 1)) - 2

    f['Y3'] = -np.arange(100).reshape(-1, 2)
    f['X4'] = np.arange(100, 200).reshape(-1, 2)

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
        for i in xrange(10):
            start, end = np.random.rand(1)[0] / 2, np.random.rand(1)[0] / 2
            while abs(start) - abs(end) < 0.1:
                start, end = np.random.rand(1)[0] / 2, np.random.rand(1)[0] / 2
            seed = 13
            # All batch is similar
            for shuffle in (True, False):
                for mode in (0, 1, 2):
                    for X_, y_ in izip(
                        self.X34.iter(batch_size=9, start=start, end=end,
                                      shuffle=shuffle, seed=seed, normalizer=None, mode=mode),
                        self.y34.iter(batch_size=9, start=start, end=end,
                                      shuffle=shuffle, seed=seed, normalizer=None, mode=mode)):
                        s = (X_ + y_).ravel().tolist()
                        self.assertEqual(s, [0.] * len(s),
                            'X and y has different order, shuffle=' + str(shuffle) + ', mode=' + str(mode))
                        s = np.sum(X_.ravel() == -y_.ravel())
                        self.assertEqual(s, np.prod(X_.shape),
                            'X and y has different order, shuffle=' + str(shuffle) + ', mode=' + str(mode))
            for shuffle in (True, False):
                for mode in (0, 1, 2):
                    for X_, y_ in izip(
                        self.X.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                    seed=seed, normalizer=None, mode=mode),
                        self.y.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                    seed=seed, normalizer=None, mode=mode)):
                        s = (X_ + y_).ravel().tolist()
                        self.assertEqual(s, [0.] * len(s),
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
        s = (x + y).ravel().tolist()
        self.assertEqual(s, [0.] * len(s)) # only case with same order

        x = np.concatenate(list(X12.iter(9, start=0, end=1., shuffle=True, mode=0)), 0).ravel()
        y = np.concatenate(list(Y12.iter(9, start=0, end=1., shuffle=True, mode=0)), 0).ravel()
        self.assertEqual(np.sum(x + y), 0.) # order is different but sum must = 0

        x = np.concatenate(list(X12.iter(9, start=0, end=1., shuffle=True, mode=2)), 0).ravel()
        y = np.concatenate(list(Y12.iter(9, start=0, end=1., shuffle=True, mode=2)), 0).ravel()
        self.assertEqual(np.sum(x + y), 0.) # order is different but sum must = 0

        # this one only support full dataset, number of element and its value
        # preserved but different order
        start, end = 0., 1.
        seed = np.random.randint(0, 10e8, 1)
        for shuffle in (False, True):
            for mode in (0, 2):
                X_ = np.concatenate(list(self.X12.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                seed=seed, normalizer=None, mode=mode)), 0)
                y_ = np.concatenate(list(self.Y.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                seed=seed, normalizer=None, mode=mode)), 0)
                self.assertEqual(np.sum(X_ + y_), 0,
                    'shuffle=' + str(shuffle) + ', mode=' + str(mode))

    def test_single_iteration(self):
        l = np.concatenate(list(self.X34.iter(batch_size=9, mode=0, shuffle=False)), 0)
        self.assertEqual(l.ravel().tolist(), range(200))

        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=0, shuffle=False, start=0., end=0.5)), 0)
        self.assertEqual(l.ravel().tolist(), range(100))

        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=0, shuffle=False, start=0.5, end=0.8)), 0)
        self.assertEqual(l.ravel().tolist(), range(100, 160))

        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=0, shuffle=True, start=0.8, end=0.5)), 0)
        self.assertEqual(sorted(l.ravel().tolist()), range(100, 160))

        # ====== Mode=2 ====== #
        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=2, shuffle=True, start=0.5, end=0.8)), 0)
        self.assertEqual(sorted(l.ravel().tolist()), range(50, 80) + range(150, 180))

        l = np.concatenate(
            list(self.X34.iter(batch_size=9, mode=2, shuffle=False, start=0.5, end=0.8)), 0)
        self.assertEqual(sorted(l.ravel().tolist()), range(50, 80) + range(150, 180))

    def test_batch_append_duplicate(self):
        try:
            f = h5py.File('test.h5', 'w')
            b = dnntoolkit.batch('test', f)
            b.append(np.arange(20).reshape(-1, 2))
            b.duplicate(3)

            b = dnntoolkit.batch(['test1', 'test2'], [f, f])
            b.append(np.arange(20, 40).reshape(-1, 2))
            b.append(np.arange(20, 40).reshape(-1, 2))
            f.close()

            f = h5py.File('test.h5', 'r')
            # duplicate
            self.assertEqual(tuple(f['test'].value.ravel().tolist()),
                             tuple(range(20) * 3))
            # multiple append
            self.assertEqual(tuple(f['test1'].value.ravel().tolist()),
                             tuple(f['test2'].value.ravel().tolist()))
            self.assertEqual(tuple(f['test1'].value.ravel().tolist()),
                             tuple(range(20, 40) * 2))
            f.close()
        except Exception, e:
            raise e
        finally:
            os.remove('test.h5')

    def test_batch_array_mode(self):
        b = dnntoolkit.batch(arrays=[
            np.arange(30, 40).reshape(-1, 2),
            np.arange(30).reshape(-1, 2)
        ])
        b.duplicate(2)
        self.assertEqual(tuple(b[:].ravel().tolist()),
                         tuple(range(30, 40) * 2 + range(30) * 2))

        b.append(np.arange(40, 50).reshape(-1, 2))
        tmp = np.asarray(
            range(30, 40) * 2 + range(40, 50) + range(30) * 2 + range(40, 50)).reshape(-1, 2)
        self.assertEqual(tuple(b.value.ravel().tolist()),
                         tuple(tmp.ravel().tolist()))

        # arithmetic test
        self.assertEqual(tmp.sum(0).tolist(), b.sum(0).tolist())
        self.assertEqual(tmp.sum(1).tolist(), b.sum(1).tolist())

        self.assertEqual(np.power(tmp, 2).sum(0).tolist(), b.sum2(0).tolist())
        self.assertEqual(np.power(tmp, 2).sum(1).tolist(), b.sum2(1).tolist())

        self.assertEqual(tmp.mean(0).tolist(), b.mean(0).tolist())
        self.assertEqual(tmp.mean(1).tolist(), b.mean(1).tolist())

        self.assertEqual(tmp.var(0).tolist(), b.var(0).tolist())
        self.assertEqual(tmp.var(1).tolist(), b.var(1).tolist())

        # Iteration test
        it = np.concatenate(list(b.iter(7, shuffle=False, mode=0)), 0)
        self.assertEqual(tuple(it.ravel().tolist()),
                         tuple(tmp.ravel().tolist()))

        it = np.concatenate(list(b.iter(7, shuffle=False, mode=2)), 0)
        self.assertEqual(np.sum(it), np.sum(tmp))
        self.assertEqual(tuple([(i, j) for i, j in sp.stats.itemfreq(it.ravel())]),
                         tuple([(i, j) for i, j in sp.stats.itemfreq(tmp.ravel())]))

        it = np.concatenate(list(b.iter(7, shuffle=True, mode=2)), 0)
        self.assertEqual(np.sum(it), np.sum(tmp))
        self.assertEqual(tuple([(i, j) for i, j in sp.stats.itemfreq(it.ravel())]),
                         tuple([(i, j) for i, j in sp.stats.itemfreq(tmp.ravel())]))

        it = np.concatenate(list(b.iter(7, shuffle=True, mode=2)), 0)
        self.assertEqual(np.sum(it), np.sum(tmp))
        self.assertEqual(tuple([(i, j) for i, j in sp.stats.itemfreq(it.ravel())]),
                         tuple([(i, j) for i, j in sp.stats.itemfreq(tmp.ravel())]))

        it = np.concatenate(list(b.iter(7, shuffle=True, mode=2, start=0.2, end=0.6)), 0)
        tmp = np.asarray(
            (range(30, 40) * 2 + range(40, 50))[6:18] +
            (range(30) * 2 + range(40, 50))[14:42]).reshape(-1, 2)
        self.assertEqual(np.sum(it), np.sum(tmp))
        self.assertEqual(tuple([(i, j) for i, j in sp.stats.itemfreq(it.ravel())]),
                         tuple([(i, j) for i, j in sp.stats.itemfreq(tmp.ravel())]))

        for i in xrange(10): # stable shape for upsample
            seed = np.random.randint(0, 10e8, 2)
            it1 = np.concatenate(
                list(b.iter(7, shuffle=True, mode=1, start=0.2, end=0.6, seed=seed[0])), 0)
            it2 = np.concatenate(
                list(b.iter(7, shuffle=True, mode=1, start=0.2, end=0.6, seed=seed[1])), 0)
            self.assertEqual(it1.shape, it2.shape)

    def test_consitent_iter_hdfBatch(self):
        try:
            f = h5py.File('test.h5', 'w')
            f['X'] = np.arange(100).reshape(-1, 2)
            f['y'] = np.arange(100, 200).reshape(-1, 2)

            f1 = h5py.File('test1.h5', 'w')
            f1['X'] = np.arange(100).reshape(-1, 2)
            f1['y'] = np.arange(100, 200).reshape(-1, 2)

            b1 = dnntoolkit.batch(['X', 'y'], [f, f1])
            b2 = dnntoolkit.batch(['X', 'y'], [f1, f])
            self.assertEqual(np.sum(b1[:] - b2[:]), 0)

            for i in xrange(10):
                start, end = np.random.rand(1)[0] / 2, np.random.rand(1)[0] / 2 + 0.5
                seed = 13
                for shuffle in (True, False):
                    for mode in (0, 1, 2):
                        x = np.concatenate(
                            list(b1.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                        seed=seed, normalizer=None, mode=mode)), 0)
                        y = np.concatenate(
                            list(b2.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                        seed=seed, normalizer=None, mode=mode)), 0)
                        s = (x - y).ravel().tolist()
                        self.assertEqual(s, [0.] * len(s),
                            'Shuffle=%s mode=%d' % (shuffle, mode))
        except Exception, e:
            raise e
        finally:
            os.remove('test.h5')
            os.remove('test1.h5')

    def test_iter_matching_arrayBatch_hdfBatch(self):
        try:
            f = h5py.File('test.h5', 'w')
            f['X'] = np.arange(100).reshape(-1, 2)

            b1 = dnntoolkit.batch('X', f)
            b2 = dnntoolkit.batch(arrays=np.arange(100).reshape(-1, 2))
            self.assertEqual(np.sum(b1[:] - b2[:]), 0)

            for i in xrange(10):
                start, end = np.random.rand(1)[0] / 2, np.random.rand(1)[0] / 2 + 0.5
                seed = 13
                for shuffle in (True, False):
                    for mode in (0, 1, 2):
                        x = np.concatenate(
                            list(b1.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                        seed=seed, normalizer=None, mode=mode)), 0)
                        y = np.concatenate(
                            list(b2.iter(batch_size=9, start=start, end=end, shuffle=shuffle,
                                        seed=seed, normalizer=None, mode=mode)), 0)
                        s = (x - y).ravel().tolist()
                        self.assertEqual(s, [0.] * len(s),
                            'Shuffle=%s mode=%d' % (shuffle, mode))
        except Exception, e:
            raise e
        finally:
            os.remove('test.h5')

    def test_imbalanced_batch_mode0(self):
        f1 = h5py.File('test1.h5', 'w')
        f1['X1'] = np.zeros((100, 1)) + 1
        f2 = h5py.File('test2.h5', 'w')
        f2['X2'] = np.zeros((5, 1)) + 2
        f2['X3'] = np.zeros((195, 1)) + 3
        f2['Y'] = np.asarray([-1] * 100 + [-2] * 5 + [-3] * 195)[:, None]
        f1.flush()
        f2.flush()
        X12 = dnntoolkit.batch(['X1', 'X2', 'X3'], [f1, f2, f2])
        Y12 = dnntoolkit.batch('Y', f2)
        s = (X12[:].ravel() + Y12[:].ravel()).tolist()
        self.assertEqual(s, [0.] * len(s))

        for shuffle in (False, True):
            for x, y in izip(X12.iter(9, start=0, end=1., shuffle=shuffle, mode=0),
                             Y12.iter(9, start=0, end=1., shuffle=shuffle, mode=0)):
                a = (x.ravel() + y.ravel()).tolist()
                self.assertEqual(a, [0.] * len(a))
            a = np.concatenate(list(X12.iter(9, start=0, end=1., shuffle=shuffle, mode=0)), 0).ravel()
            b = np.concatenate(list(Y12.iter(9, start=0, end=1., shuffle=shuffle, mode=0)), 0).ravel()
            a = (a + b).tolist()
            self.assertEqual(a, [0.] * len(a))
        os.remove('test1.h5')
        os.remove('test2.h5')

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

    def test_upsampling(self):
        try:
            # ====== Create datase ====== #
            X1 = np.arange(0, 9).reshape(-1, 3)
            X2 = np.arange(9, 24).reshape(-1, 3)
            X3 = np.arange(24, 54).reshape(-1, 3)

            Y1 = -np.arange(0, 9).reshape(-1, 3)
            Y2 = -np.arange(9, 24).reshape(-1, 3)
            Y3 = -np.arange(24, 54).reshape(-1, 3)

            f = h5py.File('test1.ds', 'w')
            f['X1'] = X1
            f['Y1'] = Y1
            f.close()
            f = h5py.File('test2.ds', 'w')
            f['X2'] = X2
            f['Y2'] = Y2
            f.close()
            f = h5py.File('test3.ds', 'w')
            f['X3'] = X3
            f['Y3'] = Y3
            f.close()
            # ====== Test ====== #
            ds = dnntoolkit.dataset(['test1.ds', 'test2.ds', 'test3.ds'], 'r')
            X = ds[['X1', 'X2', 'X3']]
            Y = ds[['Y1', 'Y2', 'Y3']]

            # check shape
            self.assertEqual(X.shape, Y.shape)

            # check order
            tmp = (X[:] + Y[:]).ravel().tolist()
            self.assertEqual(tmp, [0.] * len(tmp))

            # check order
            tmp = (np.concatenate(list(X.iter(7, 0., 1., True, mode=0)), 0) +
                   np.concatenate(list(Y.iter(7, 0., 1., True, mode=0)), 0)).ravel().tolist()
            self.assertEqual(tmp, [0.] * len(tmp))

            tmp = (np.concatenate(list(X.iter(7, 0., 1., True, mode=1)), 0) +
                   np.concatenate(list(Y.iter(7, 0., 1., True, mode=1)), 0)).ravel().tolist()
            self.assertEqual(tmp, [0.] * len(tmp))

            tmp = (np.concatenate(list(X.iter(7, 0., 1., True, mode=2)), 0) +
                   np.concatenate(list(Y.iter(7, 0., 1., True, mode=2)), 0)).ravel().tolist()
            self.assertEqual(tmp, [0.] * len(tmp))

            tmp = []
            for i, j in izip(X.iter(7, 0., 1., True, mode=0),
                             Y.iter(7, 0., 1., True, mode=0)):
                tmp += (i + j).ravel().tolist()
            self.assertEqual(tmp, [0.] * len(tmp))

            tmp = []
            for i, j in izip(X.iter(7, 0., 1., True, mode=1),
                             Y.iter(7, 0., 1., True, mode=1)):
                tmp += (i + j).ravel().tolist()
            self.assertEqual(tmp, [0.] * len(tmp))

            tmp = []
            for i, j in izip(X.iter(7, 0., 1., True, mode=2),
                             Y.iter(7, 0., 1., True, mode=2)):
                tmp += (i + j).ravel().tolist()
            self.assertEqual(tmp, [0.] * len(tmp))
        except Exception, e:
            import traceback; traceback.print_exc();
            raise e
        finally:
            if os.path.exists('test1.ds'):
                os.remove('test1.ds')
            if os.path.exists('test2.ds'):
                os.remove('test2.ds')
            if os.path.exists('test3.ds'):
                os.remove('test3.ds')
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
