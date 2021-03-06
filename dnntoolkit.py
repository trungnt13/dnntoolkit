# coding=utf-8
# Copyright 2015 Trung Ngo Trong

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
# Credits: Richard Kurle

# ======================================================================
# Collection of utilities function in python for deep learning development
# Library
# * numpy
# * scipy
# * theano
# * sidekit
# * paramiko
# * soundfile
# * h5py
# * pandas
# * numba
# ======================================================================
from __future__ import print_function, division

import os
import sys
import math
import random
import time
import warnings
import types
from stat import S_ISDIR

import itertools
import six
from six.moves import zip, zip_longest
from collections import OrderedDict, defaultdict

import numpy as np
from numpy.random import RandomState
import scipy as sp

import theano
from theano import tensor as T

import h5py
import pandas as pd
import soundfile
import paramiko

MAGIC_SEED = 12082518

# ======================================================================
# Helper data structure
# ======================================================================
class queue(object):

    """ FIFO, fast, NO thread-safe queue """

    def __init__(self):
        super(queue, self).__init__()
        self._data = []
        self._idx = 0

    def put(self, value):
        self._data.append(value)

    def append(self, value):
        self._data.append(value)

    def pop(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def get(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def empty(self):
        if self._idx == len(self._data):
            return True
        return False

    def clear(self):
        del self._data
        self._data = []
        self._idx = 0

    def __len__(self):
        return len(self._data) - self._idx

# ======================================================================
# Multiprocessing
# ======================================================================
class mpi():

    """docstring for mpi"""
    @staticmethod
    def segment_job(file_list, n_seg):
        '''
        Example
        -------
            >>> segment_job([1,2,3,4,5],2)
            >>> [[1, 2, 3], [4, 5]]
            >>> segment_job([1,2,3,4,5],4)
            >>> [[1], [2], [3], [4, 5]]
        '''
        # by floor, make sure and process has it own job
        size = int(np.ceil(len(file_list) / float(n_seg)))
        if size * n_seg - len(file_list) > size:
            size = int(np.floor(len(file_list) / float(n_seg)))
        # start segmenting
        segments = []
        for i in xrange(n_seg):
            start = i * size
            if i < n_seg - 1:
                end = start + size
            else:
                end = max(start + size, len(file_list))
            segments.append(file_list[start:end])
        return segments

    @staticmethod
    def div_n_con(path, file_list, n_job, div_func, con_func):
        ''' Divide and conquer strategy for multiprocessing.
        Parameters
        ----------
        path : str
            path to save the result, all temp file save to path0, path1, path2...
        file_list : list
            list of all file or all job to do processing
        n_job : int
            number of processes
        div_func : function(save_path, jobs_list)
            divide function, execute for each partition of job
        con_func : function(save_path, temp_paths)
            function to merge all the result
        Returns
        -------
        return : list(Process)
            div_processes and con_processes
        '''
        import multiprocessing
        job_list = mpi.segment_job(file_list, n_job)
        file_path = [path + str(i) for i in xrange(n_job)]
        div_processes = [multiprocessing.Process(target=div_func, args=(file_path[i], job_list[i])) for i in xrange(n_job)]
        con_processes = multiprocessing.Process(target=con_func, args=(path, file_path))
        return div_processes, con_processes

    @staticmethod
    def preprocess_mpi(jobs_list, features_func, save_func, n_cache=30, log_point=50):
        ''' Wrapped preprocessing procedure in MPI.
                    root
                / / / | \ \ \
                features_func
                \ \ \ | / / /
                  save_func
            * NO need call Barrier at the end of this methods

        Parameters
        ----------
        jobs_list : list
            [data_concern_job_1, job_2, ....]
        features_func : function(job_i)
            function object to extract feature from each job
        save_func : function([job_i,...])
            transfer all data to process 0 as a list for saving to disk
        n_cache : int
            maximum number of cache for each process before gathering the data
        log_point : int
            after this amount of preprocessed data, print log

        Notes
        -----
        Any None return by features_func will be ignored

        Example
        -------
        >>> jobs = range(1, 110)
        >>> if rank == 0:
        >>>     f = h5py.File('tmp.hdf5', 'w')
        >>>     idx = 0
        >>> def feature_extract(j):
        >>>     return rank
        >>> def save(j):
        >>>     global idx
        >>>     f[str(idx)] = str(j)
        >>>     idx += 1
        >>> dnntoolkit.mpi.preprocess_mpi(jobs, feature_extract, save, n_cache=5)
        >>> if rank == 0:
        >>>     f['idx'] = idx
        >>>     f.close()
        '''
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        npro = comm.Get_size()

        #####################################
        # 1. Scatter jobs for all process.
        if rank == 0:
            logger.info('Process 0 found %d jobs' % len(jobs_list))
            jobs = mpi.segment_job(jobs_list, npro)
            n_loop = max([len(i) for i in jobs])
        else:
            jobs = None
            n_loop = 0
            logger.info('Process %d waiting for Process 0!' % rank)
        comm.Barrier()

        jobs = comm.scatter(jobs, root=0)
        n_loop = comm.bcast(n_loop, root=0)
        logger.info('Process %d receive %d jobs' % (rank, len(jobs)))

        #####################################
        # 2. Start preprocessing.
        data = []

        for i in xrange(n_loop):
            if i % n_cache == 0 and i > 0:
                all_data = comm.gather(data, root=0)
                if rank == 0:
                    logger.info('Saving data at process 0')
                    all_data = [k for j in all_data for k in j]
                    if len(all_data) > 0:
                        save_func(all_data)
                data = []

            if i >= len(jobs): continue
            feature = features_func(jobs[i])
            if feature is not None:
                data.append(feature)

            if i % log_point == 0:
                logger.info('Rank:%d preprocessed %d files!' % (rank, i))

        all_data = comm.gather(data, root=0)
        if rank == 0:
            logger.info('Finished preprocess_mpi !!!!\n')
            all_data = [k for j in all_data for k in j]
            if len(all_data) > 0:
                save_func(all_data)


# ======================================================================
# io helper
# ======================================================================
class io():

    @staticmethod
    def all_files(path, filter_func=None):
        ''' Recurrsively get all files in the given path '''
        file_list = []
        q = queue()
        # init queue
        if os.access(path, os.R_OK):
            for p in os.listdir(path):
                q.put(os.path.join(path, p))
        # process
        while not q.empty():
            p = q.pop()
            if os.path.isdir(p):
                if os.access(p, os.R_OK):
                    for i in os.listdir(p):
                        q.put(os.path.join(p, i))
            else:
                if filter_func is not None and not filter_func(p):
                    continue
                file_list.append(p)
        return file_list

    @staticmethod
    def get_from_module(module, identifier, environment=None):
        '''
        Parameters
        ----------
        module : ModuleType, str
            module contain the identifier
        identifier : str
            str, name of identifier
        environment : map
            map from globals() or locals()

        Returns
        -------
        object : with the same name as identifier
        None : not found
        '''
        if isinstance(module, six.string_types):
            if environment and module in environment:
                module = environment[module]
            elif module in globals():
                module = globals()[module]
            else:
                return None

        from inspect import getmembers
        for i in getmembers(module):
            if identifier in i:
                return i[1]
        return None

    @staticmethod
    def search_id(identifier, prefix='', suffix='', path='.', exclude='',
                  prefer_compiled=False):
        ''' Algorithms:
         - Search all files in the `path` matched `prefix` and `suffix`
         - Exclude all files contain any str in `exclude`
         - Sorted all files based on alphabet
         - Load all modules based on `prefer_compiled`
         - return list of identifier found in all modules

        Parameters
        ----------
        identifier : str
            identifier of object, function or anything in script files
        prefix : str
            prefix of file to search in the `path`
        suffix : str
            suffix of file to search in the `path`
        path : str
            searching path of script files
        exclude : str, list(str)
            any files contain str in this list will be excluded
        prefer_compiled : bool
            True mean prefer .pyc file, otherwise prefer .py

        Returns
        -------
        list(object, function, ..) :
            any thing match given identifier in all found script file

        Notes
        -----
        File with multiple . character my procedure wrong results
        If the script run this this function match the searching process, a
        infinite loop may happen!
        '''
        import re
        import imp
        from inspect import getmembers
        # ====== validate input ====== #
        if exclude == '': exclude = []
        if type(exclude) not in (list, tuple, np.ndarray):
            exclude = [exclude]
        prefer_flag = -1
        if prefer_compiled: prefer_flag = 1
        # ====== create pattern and load files ====== #
        pattern = re.compile('^%s.*%s\.pyc?' % (prefix, suffix)) # py or pyc
        files = os.listdir(path)
        files = [f for f in files
                 if pattern.match(f) and
                 sum([i in f for i in exclude]) == 0]
        # ====== remove duplicated pyc files ====== #
        files = sorted(files, key=lambda x: prefer_flag * len(x)) # pyc is longer
        # .pyc go first get overrided by .py
        files = sorted({f.split('.')[0]:f for f in files}.values())

        # ====== load all modules ====== #
        modules = []
        for f in files:
            try:
                if '.pyc' in f:
                    modules.append(
                            imp.load_compiled(f.split('.')[0],
                                              os.path.join(path, f))
                        )
                else:
                    modules.append(
                            imp.load_source(f.split('.')[0],
                                            os.path.join(path, f))
                        )
            except:
                pass
        # ====== Find all identifier in modules ====== #
        ids = []
        for m in modules:
            for i in getmembers(m):
                if identifier in i:
                    ids.append(i[1])
        # remove duplicate py
        return ids

# ===========================================================================
# Statistic
# ===========================================================================
class stats():
    pass

# ======================================================================
# Array Utils
# ======================================================================
class tensor():
    # ==================== Sequence processing ==================== #

    @staticmethod
    def pad_sequences(sequences, maxlen=None, dtype='int32',
                      padding='pre', truncating='pre', value=0.):
        '''
        Pad each sequence to the same length:
        the length of the longest sequence.

        If maxlen is provided, any sequence longer than maxlen is truncated
        to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.

        Supports post-padding and pre-padding (default).

        Parameters:
        -----------
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger than
                maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.

        Returns:
        -------
        x: numpy array with dimensions (number_of_sequences, maxlen)

        Example:
        -------
            > pad_sequences([[1,2,3],
                             [1,2],
                             [1,2,3,4]], maxlen=3, padding='post', truncating='pre')
            > [[1,2,3],
               [1,2,0],
               [2,3,4]]
        '''
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue # empty list was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % padding)

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return x

    @staticmethod
    def masked_output(X, X_mask):
        '''
        Example
        -------
            X: [[1,2,3,0,0],
                [4,5,0,0,0]]
            X_mask: [[1,2,3,0,0],
                     [4,5,0,0,0]]
            return: [[1,2,3],[4,5]]
        '''
        res = []
        for x, mask in zip(X, X_mask):
            x = x[np.nonzero(mask)]
            res.append(x.tolist())
        return res

    @staticmethod
    def to_categorical(y, n_classes=None):
        '''Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy
        '''
        y = np.asarray(y, dtype='int32')
        if not n_classes:
            n_classes = np.max(y) + 1
        Y = np.zeros((len(y), n_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y

    @staticmethod
    def split_chunks(a, maxlen, overlap):
        '''
        Example
        -------
        >>> print(split_chunks(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 5, 1))
        >>> [[1, 2, 3, 4, 5],
               [4, 5, 6, 7, 8]]
        '''
        chunks = []
        nchunks = int((max(a.shape) - maxlen) / (maxlen - overlap)) + 1
        for i in xrange(nchunks):
            start = i * (maxlen - overlap)
            chunks.append(a[start: start + maxlen])

        # ====== Some spare frames at the end ====== #
        wasted = max(a.shape) - start - maxlen
        if wasted >= (maxlen - overlap) / 2:
            chunks.append(a[-maxlen:])
        return chunks

    @staticmethod
    def set_ordered(seq):
       seen = {}
       result = []
       for marker in seq:
           if marker in seen: continue
           seen[marker] = 1
           result.append(marker)
       return result

    @staticmethod
    def shrink_labels(labels, maxdist=1):
        '''
        Example
        -------
        >>> print(shrink_labels(np.array([0, 0, 1, 0, 1, 1, 0, 0, 4, 5, 4, 6, 6, 0, 0]), 1))
        >>> [0, 1, 0, 1, 0, 4, 5, 4, 6, 0]
        >>> print(shrink_labels(np.array([0, 0, 1, 0, 1, 1, 0, 0, 4, 5, 4, 6, 6, 0, 0]), 2))
        >>> [0, 1, 0, 4, 6, 0]

        Notes
        -----
        Different from set_ordered, the resulted array still contain duplicate
        if they a far away each other.
        '''
        maxdist = max(1, maxdist)
        out = []
        l = len(labels)
        i = 0
        while i < l:
            out.append(labels[i])
            last_val = labels[i]
            dist = min(maxdist, l - i - 1)
            j = 1
            while (i + j < l and labels[i + j] == last_val) or (j < dist):
                j += 1
            i += j
        return out

    # ==================== theano ==================== #

    @staticmethod
    def floatX(X):
        return np.asarray(X, dtype=theano.config.floatX)

    @staticmethod
    def sharedX(X, dtype=theano.config.floatX, name=None):
        return theano.shared(np.asarray(X, dtype=dtype), name=name)

    @staticmethod
    def shared_zeros(shape, dtype=theano.config.floatX, name=None):
        return T.sharedX(np.zeros(shape), dtype=dtype, name=name)

    @staticmethod
    def shared_scalar(val=0., dtype=theano.config.floatX, name=None):
        return theano.shared(np.cast[dtype](val), name=name)

    @staticmethod
    def shared_ones(shape, dtype=theano.config.floatX, name=None):
        return T.sharedX(np.ones(shape), dtype=dtype, name=name)

    @staticmethod
    def alloc_zeros_matrix(*dims):
        return T.alloc(np.cast[theano.config.floatX](0.), *dims)

    @staticmethod
    def ndim_tensor(ndim):
        if ndim == 1:
            return T.vector()
        elif ndim == 2:
            return T.matrix()
        elif ndim == 3:
            return T.tensor3()
        elif ndim == 4:
            return T.tensor4()
        return T.matrix()

    @staticmethod
    def one_hot_max(x):
        '''
        Example
        -------
        >>> Input: [[0.0, 0.0, 0.5],
        >>>         [0.0, 0.3, 0.1],
        >>>         [0.6, 0.0, 0.2]]
        >>> Output: [[0.0, 0.0, 1.0],
        >>>         [0.0, 1.0, 0.0],
        >>>         [1.0, 0.0, 0.0]]
        '''
        return T.cast(T.eq(T.arange(x.shape[1])[None, :], T.argmax(x, axis=1, keepdims=True)), theano.config.floatX)

    def apply_mask(x, mask):
        '''
        Example
        -------
        >>> Input: [128,500,120]
        >>> Mask:  [128,500]
        >>> Output: [128,500,120]
        '''
        return T.mul(x, T.DimShuffle(mask, (0,1,'x')))
# ======================================================================
# Computing hyper-parameters
# ======================================================================
# remember: NOT all GPU resources is only used for your task
class GPU():
    K40 = {
        'performance': 5, # Tflops (GPU Boost Clocks)
        'bandwidth': 288, # GB/sec
        'size': 12, # GB
        'core': 2880
    }
    K80 = {
        'performance': 8.74, # Tflops (GPU Boost Clocks)
        'bandwidth': 480, # GB/sec
        'size': 24, # GB
        'core': 4992
    }

    @staticmethod
    def on_gpu():
        return theano.config.device[:3] == 'gpu'

    @staticmethod
    def check_cudnn():
        from theano.sandbox.cuda.dnn import dnn_available as d
        logger.debug(d() or d.msg)

# ===========================================================================
# DNN utilities
# ===========================================================================
def _check_gs(validation):
    ''' Generalization sensitive:
    validation is list of cost values (assumpt: lower is better)
    '''
    if len(validation) == 0:
        return 0, 0
    shouldStop = 0
    shouldSave = 0

    if validation[-1] > min(validation):
        shouldStop = 1
        shouldSave = -1
    else:
        shouldStop = -1
        shouldSave = 1

    return shouldSave, shouldStop

def _check_gl(validation, threshold=5):
    ''' Generalization loss:
    validation is list of cost values (assumpt: lower is better)
    Note
    ----
    This strategy prefer to keep the model remain when the cost unchange
    '''
    gl_exit_threshold = threshold
    epsilon = 1e-5

    if len(validation) == 0:
        return 0, 0
    shouldStop = 0
    shouldSave = 0

    gl_t = 100 * (validation[-1] / (min(validation) + epsilon) - 1)
    if gl_t <= 0 and np.argmin(validation) == (len(validation) - 1):
        shouldSave = 1
        shouldStop = -1
    elif gl_t > gl_exit_threshold:
        shouldStop = 1
        shouldSave = -1

    # check stay the same performance for so long
    if len(validation) > threshold:
        remain_detected = 0
        j = validation[-int(threshold)]
        for i in validation[-int(threshold):]:
            if abs(i - j) < epsilon:
                remain_detected += 1
        if remain_detected >= threshold:
            shouldStop = 1
    return shouldSave, shouldStop


def _check_hope_and_hop(validation):
    ''' Hope and hop:
    validation is list of cost values (assumpt: lower is better)
    '''
    patience = 5
    patience_increase = 0.5
    improvement_threshold = 0.998
    if len(validation) == 0:
        return 0, 0
    shouldStop = 0
    shouldSave = 0

    # one more iteration
    i = len(validation)
    if len(validation) == 1: # cold start
        shouldSave = 1
        shouldStop = -1
    else: # warm up
        last_best_validation = min(validation[:-1])
        # significant improvement
        if min(validation) < last_best_validation * improvement_threshold:
            patience += i * patience_increase
            shouldSave = 1
            shouldStop = -1
        # punish
        else:
            # the more increase the faster we running out of patience
            rate = validation[-1] / last_best_validation
            patience -= i * patience_increase * rate
            # if still little bit better, just save it
            if min(validation) < last_best_validation:
                shouldSave = 1
            else:
                shouldSave = -1

    if patience <= 0:
        shouldStop = 1
        shouldSave = -1
    return shouldSave, shouldStop

class dnn():

    @staticmethod
    def earlystop(costs, generalization_loss = False,
        generalization_sensitive=False, hope_hop=False,
        threshold=None):
        ''' Early stop.

        Parameters
        ----------
        generalization_loss : type
            note
        generalization_sensitive : type
            note
        hope_hop : type
            note

        Returns
        -------
        return : boolean, boolean
            shouldSave, shouldStop

        '''
        values = costs
        shouldSave = 0
        shouldStop = 0
        if generalization_loss:
            if threshold is not None:
                save, stop = _check_gl(values, threshold)
            else:
                save, stop = _check_gl(values)
            shouldSave += save
            shouldStop += stop
        if generalization_sensitive:
            save, stop = _check_gs(values)
            shouldSave += save
            shouldStop += stop
        if hope_hop:
            save, stop = _check_hope_and_hop(values)
            shouldSave += save
            shouldStop += stop
        return shouldSave > 0, shouldStop > 0

    @staticmethod
    def check_weights(weights):
        raise NotImplementedError()

    @staticmethod
    def check_gradients(gradients, epsilon=10e-8, threshold=10e4):
        '''
        Returns
        -------
        False : nothing wrong with gradients
        1 : Nan in gradients
        2 : 25 percentage of gradients ~ 0.
        3 : gradients is exploded
        '''
        first_grads = None
        last_grads = None

        # ====== only need to check the first gradients ====== #
        for g in gradients:
            if g.ndim >= 2:
                first_grads = g
                break
        if first_grads is None:
            first_grads = gradients[0]

        # ====== get statistic of grads ====== #
        # NaN gradients
        if np.isnan(np.min(first_grads)):
            return 1
        # too small value: Vanishing
        if np.abs(np.percentile(first_grads, 0.25)) < 0. + epsilon:
            return 2

        # ====== Compare to last gradients ====== #
        for g in reversed(gradients):
            if g.ndim >= 2:
                last_grads = g
                break
        if last_grads is None:
            last_grads = gradients[-1]

        # exploding
        if np.mean(last_grads) / np.mean(first_grads) > threshold:
            return 3
        return False

    @staticmethod
    def est_weights_decay(nparams, maxval=0.1):
        '''
            10^{log10(1/sqrt(nparams))}
        '''
        l2value = min(10**np.log10(1. / nparams**(1 / 2.4)),
                      maxval)
        return np.cast[theano.config.floatX](l2value / 2)

    @staticmethod
    def est_maxnorm(nparams, maxval=1000.):
        '''
            10^{log10(1/sqrt(nparams))}
        '''
        pivot = nparams*10e-5
        log10 = 10**int(np.log10(pivot))
        point = np.ceil(pivot / log10)
        return min(maxval, np.cast[theano.config.floatX](point * log10))

    @staticmethod
    def est_lr(nparams, nlayers, maxval=0.1):
        '''
            10^{log10(1/sqrt(nparams))}
        '''
        nparams *= np.sqrt(nlayers)
        # very heuristic value
        l2value = min(10**np.log10(1. / nparams**(1/2.03)), maxval)
        return np.cast[theano.config.floatX](l2value)

    @staticmethod
    def calc_lstm_weights(n_in, n_units, peephole=True):
        b = n_units * 3
        w_in = n_in * n_units * 4
        w_hid = n_units * n_units * 4
        w_cell = 0
        if peephole:
            w_cell = n_units * n_units * 3
            b += n_units
        return b + w_in + w_hid + w_cell
# ===========================================================================
# Model
# ===========================================================================
def _get_ms_time():
    return int(round(time.time() * 1000)) # in ms

class _history(object):

    """
    Simple object to record data row in form:
        [time, [tags], values]

    Notes
    -----
    Should store primitive data
    """

    def __init__(self, name=None, description=None):
        super(_history, self).__init__()
        self._name = name
        self._description = description
        self._history = []
        self._init_time = _get_ms_time()

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def time(self):
        return self._init_time

    # ==================== multiple history ==================== #
    def merge(self, *history):
        h = _history()
        all_names = [self.name] + [i.name for i in history]
        h._name = 'Merge:<' + ','.join([str(i) for i in all_names]) + '>'
        data = []
        data += self._history
        for i in history:
            data += i._history
        data = sorted(data, key=lambda x: x[0])
        h._history = data
        return h

    # ==================== History manager ==================== #

    def clear(self):
        self._history = []

    def record(self, values, *tags):
        curr_time = _get_ms_time()

        if not isinstance(tags, list) and not isinstance(tags, tuple):
            tags = [tags]
        tags = set(tags)

        # timestamp must never equal
        if len(self._history) > 0 and self._history[-1][0] >= curr_time:
            curr_time = self._history[-1][0] + 1

        self._history.append([curr_time, tags, values])

    def update(self, tags, new, after=None, before=None, n=None, absolute=False):
        ''' Apply a funciton to all selected value

        Parameters
        ----------
        tags : list, str, filter function or any comparable object
            get all values contain given tags
        new : function, object, values
            update new values
        after, before : time constraint (in millisecond)
            after < t < before
        n : int
            number of record will be update
        filter_value : function
            function to filter each value found
        absolute : boolean
            whether required the same set of tags or just contain

        '''
        # ====== preprocess arguments ====== #
        history = self._history
        if not isinstance(tags, list) and not isinstance(tags, tuple):
            tags = [tags]
        tags = set(tags)
        tags = [t if hasattr(t, '__call__') else _create_comparator(t) for t in tags]

        if len(history) == 0:
            return []
        if not hasattr(tags, '__len__'):
            tags = [tags]
        if after is None:
            after = history[0][0]
        if before is None:
            before = history[-1][0]
        if n is None or n < 0:
            n = len(history)

        # ====== searching ====== #
        count = 0
        for row in history:
            if count > n:
                break

            # check time
            time = row[0]
            if time < after or time > before:
                continue
            # check tags
            if not _is_tags_match(tags, row[1], absolute):
                continue
            # check value
            if hasattr(new, '__call__'):
                row[2] = new(row[2])
            else:
                row[2] = new
            count += 1

    def select(self, tags, default=None, after=None, before=None, n=None,
        filter_value=None, absolute=False, newest=False, return_time=False):
        ''' Query in history

        Parameters
        ----------
        tags : list, str, filter function or any comparable object
            get all values contain given tags
        default : object
            default return value of found nothing
        after, before : time constraint (in millisecond)
            after < t < before
        n : int
            number of record return
        filter_value : function(value)
            function to filter each value found
        absolute : boolean
            whether required the same set of tags or just contain
        newest : boolean
            returning order (newest first, default is False)
        time : boolean
            whether return time tags

        Returns
        -------
        return : list
            always return list, in case of no value, return empty list
        '''
        # ====== preprocess arguments ====== #
        history = self._history
        if not isinstance(tags, list) and not isinstance(tags, tuple):
            tags = [tags]
        tags = set(tags)
        tags = [t if hasattr(t, '__call__') else _create_comparator(t) for t in tags]

        if len(history) == 0:
            return []
        if after is None:
            after = history[0][0]
        if before is None:
            before = history[-1][0]
        if n is None or n < 0:
            n = len(history)

        # ====== searching ====== #
        res = []
        for row in history:
            if len(res) > n:
                break

            # check time
            time = row[0]
            if time < after or time > before:
                continue
            # check tags
            if not _is_tags_match(tags, row[1], absolute):
                continue
            # check value
            val = row[2]
            if filter_value is not None and not filter_value(val):
                continue
            # ok, found it!
            res.append((time, val))

        # ====== return results ====== #
        if not return_time:
            res = [i[1] for i in res]

        if newest:
            return list(reversed(res))

        if len(res) == 0 and default is not None:
            return default
        return res

    # ==================== pretty print ==================== #

    def print_history(self):
        fmt = '|%13s | %40s | %20s|'
        sep = ('-' * 13, '-' * 40, '-' * 20)
        # header
        logger.log(fmt % sep)
        logger.log(fmt % ('Time', 'Tags', 'Values'))
        logger.log(fmt % sep)
        # contents
        for row in self._history:
            row = tuple([str(i) for i in row])
            logger.log(fmt % row)

    def __str__(self):
        from collections import defaultdict
        unique_tags = defaultdict(int)
        for i in self._history:
            for j in i[1]:
                unique_tags[j] += 1
        s = ''
        s += '======== History Frame ========' + '\n'
        s += ' - Name: %s' % self._name + '\n'
        s += ' - Description: %s' % self._description + '\n'
        s += ' - Init time: %s' % time.ctime(int(self._init_time / 1000)) + '\n'
        s += ' - Statistic:' + '\n'
        s += '   - Len:%d' % len(self._history) + '\n'
        for k, v in unique_tags.iteritems():
            s += '   - tags: %-10s  -  freq:%d' % (k, v) + '\n'
        return s

def _create_comparator(t):
    return lambda x: x == t

def _is_tags_match(func, tags, absolute=False):
    '''
    Example
    -------
        > tags = [1, 2, 3]
        > func = [lambda x: x == 1]
        > func1 = [lambda x: x == 1, lambda x: x == 2, lambda x: x == 3]
        > _is_tags_match(func, tags, absolute=False) # True
        > _is_tags_match(func, tags, absolute=True) # False
        > _is_tags_match(func1, tags, absolute=True) # True
    '''
    for f in func:
        match = False
        for t in tags:
            match |= f(t)
        if not match: return False
    if absolute and len(func) != len(tags):
        return False
    return True

def serialize_sandbox(environment):
    '''environment : globals(), locals()
    Returns
    -------
    dictionary : cPickle dumps-able dictionary to store as text
    '''
    import re
    sys_module = re.compile('__\w+__')
    primitive = (bool, int, float, str,
                 tuple, list, dict, type, types.ModuleType)
    ignore_key = ['__name__', '__file__']
    sandbox = {}
    for k, v in environment.iteritems():
        if k in ignore_key: continue
        if type(v) in primitive and sys_module.match(k) is None:
            if isinstance(v, types.ModuleType):
                v = {'name': v.__name__, '__module': True}
            sandbox[k] = v

    return sandbox

def deserialize_sandbox(sandbox):
    '''
    environment : dictionary
        create by `serialize_sandbox`
    '''
    if not isinstance(sandbox, dict):
        raise ValueError(
            '[environment] must be dictionary created by serialize_sandbox')
    import importlib
    primitive = (bool, int, float, str,
                 tuple, list, dict, type, types.ModuleType)
    environment = {}
    for k, v in sandbox.iteritems():
        if type(v) in primitive:
            if isinstance(v, dict) and '__module' in v:
                v = importlib.import_module(v['name'])
            environment[k] = v
    return environment

class model(object):

    """docstring for Model
    """

    def __init__(self, savepath=None):
        super(model, self).__init__()
        self._history = []
        self._working_history = None
        self._history_updated = False

        self._weights = []
        self._save_path = savepath

        self._model_func = None
        self._model_name = None
        self._model_args = None
        self._api = ''
        self._sandbox = ''

        # contain real model object
        self._model = None
        self._pred = None

    # ==================== Weights ==================== #
    def set_weights(self, weights):
        '''
        weights : list(np.ndarray)
            list of all numpy array contain parameters

        Notes
        -----
        - if set_model is called, this methods will set_weights for your AI also
        '''
        # fetch new weights
        self._weights = []
        for w in weights:
            self._weights.append(w.astype(np.float32))
        # set model weights
        if self._model is not None:
            try:
                if self._api == 'lasagne':
                    import lasagne
                    lasagne.layers.set_all_param_values(
                        self._model, self._weights)
                else:
                    logger.warning('Need support for new API %s' % self._api)
            except Exception, e:
                logger.error('Unable to set weights for AI: %s' % str(e))

    def get_weights(self):
        ''' if set_model is called, and your AI is created, always return the
        newest weights from AI
        '''
        if self._model is not None:
            if self._api == 'lasagne':
                import lasagne
                self._weights = lasagne.layers.get_all_param_values(self._model)
            else:
                logger.warning('Need support for new API %s' % self._api)
        return self._weights

    def get_nweights(self):
        ''' Return number of weights '''
        n = 0
        weights = self.get_weights()
        for w in weights:
            n += np.prod(w.shape)
        return n

    def get_nlayers(self):
        ''' Return number of layers (only layers with trainable params) '''
        if self._model is not None:
            if self._api == 'lasagne':
                import lasagne
                return len([i for i in lasagne.layers.get_all_layers(self._model)
                            if len(i.get_params(trainable=True)) > 0])
            else:
                logger.warning('Need support for new API %s' % self._api)
        return 0

    # ==================== Network manipulation ==================== #
    def set_pred(self, pred_func):
        self._pred = pred_func

    def get_model(self):
        return self._model_func

    def set_model(self, model, api, **kwargs):
        ''' Save a function that create your model.

        Parameters
        ----------
        model : __call__ object
            main function that create your model
        api : str
            support: lasagne | keras | blocks
        kwargs : **
            any arguments for your model creatation function
        '''
        if not hasattr(model, '__call__'):
            raise NotImplementedError('Model must be a function return computational graph')
        sandbox = {}
        # ====== Lasagne ====== #
        if 'lasagne' in api:
            self._api = 'lasagne'
            self._model_func = model
            # store all primitive and api type
            sandbox = serialize_sandbox(model.func_globals)
        # ====== Keras ====== #
        elif 'keras' in api:
            self._api = 'keras'
            warnings.warn('NOT support API!', RuntimeWarning)
        elif 'blocks' in api:
            self._api = 'blocks'
            warnings.warn('NOT support API!', RuntimeWarning)
        #H5py not support binary string, cannot use marshal
        self._sandbox = sandbox
        self._model_args = kwargs
        self._model_name = model.func_name

    def create_model(self, checkpoint=True):
        '''
        Parameters
        ----------
        checkpoint: bool
            if True, not only create new model but also create a saved
            checkpoint if NO weights have setted

        Notes
        -----
        The logic of this method is:
         - if already set_weights, old weights will be loaded into new model
         - if NO setted weights and new model creates, fetch all models weights
         and save it to file to create checkpoint of save_path available
        '''
        if self._model_func is None:
            raise ValueError("You must set_model first")
        if self._model is None:
            func = self._model_func
            args = self._model_args
            logger.critical('*** creating network ... ***')
            self._model = func(**args)

            # load old weight
            if len(self._weights) > 0:
                try:
                    if self._api == 'lasagne':
                        import lasagne
                        lasagne.layers.set_all_param_values(self._model,
                                                            self._weights)
                        logger.critical('*** Successfully load old weights ***')
                    else:
                        warnings.warn('NOT support API!', RuntimeWarning)
                except Exception, e:
                    logger.critical('*** Cannot load old weights ***')
                    logger.error(str(e))
                    import traceback; traceback.print_exc();
            # fetch new weights into model,  create checkpoints
            else:
                weights = self.get_weights()
                if self._save_path is not None and checkpoint:
                    f = h5py.File(self._save_path, mode='a')
                    try:
                        if 'nb_weights' in f: del f['nb_weights']
                        f['nb_weights'] = len(weights)
                        for i, w in enumerate(weights):
                            if 'weight_%d' % i in f: del f['weight_%d' % i]
                            f['weight_%d' % i] = w
                    except Exception, e:
                        raise e
                    f.close()
        return self._model

    def create_pred(self):
        ''' Create prediction funciton '''
        self.create_model()

        # ====== Create prediction function ====== #
        if self._pred is None:
            if self._api == 'lasagne':
                import lasagne
                # create prediction function
                self._pred = theano.function(
                    inputs=[l.input_var for l in
                        lasagne.layers.find_layers(
                            self._model, types=lasagne.layers.InputLayer)],
                    outputs=lasagne.layers.get_output(
                        self._model, deterministic=True),
                    allow_input_downcast=True,
                    on_unused_input=None)
                logger.critical('*** Successfully create prediction function ***')
            else:
                warnings.warn('NOT support API!', RuntimeWarning)

    def pred(self, *X):
        '''
        Order of input will be keep in the same order when you create network
        '''
        self.create_pred()

        # ====== Check ====== #
        if self._pred is None:
            return None

        # ====== make prediction ====== #
        prediction = None
        try:
            prediction = self._pred(*X)
        except Exception, e:
            logger.critical('*** Cannot make prediction ***')
            if self._api == 'lasagne':
                import lasagne
                input_layers = lasagne.layers.find_layers(self._model, types=lasagne.layers.InputLayer)
                logger.debug('Input order:' + str([l.name for l in input_layers]))
            logger.error(str(e))
            import traceback; traceback.print_exc();
        return prediction

    def rollback(self):
        ''' Roll-back weights and history of model from last checkpoints
        (last saved path).
        '''
        if self._save_path is not None and os.path.exists(self._save_path):
            import cPickle
            f = h5py.File(self._save_path, 'r')

            # rollback weights
            if 'nb_weights' in f:
                self._weights = []
                for i in xrange(f['nb_weights'].value):
                    self._weights.append(f['weight_%d' % i].value)
            if self._model is not None:
                if self._api == 'lasagne':
                    import lasagne
                    lasagne.layers.set_all_param_values(
                        self._model, self._weights)
                    logger.critical(' *** Weights rolled-back! ***')
                else:
                    warnings.warn('NOT support API-%s!' % self._api, RuntimeWarning)

            # rollback history
            if 'history' in f:
                self._history = cPickle.loads(f['history'].value)
                logger.critical(' *** History rolled-back! ***')
                self._history_updated = True
        else:
            logger.warning('No checkpoint found! Ignored rollback!')
        return self

    # ==================== History manager ==================== #
    def _check_current_working_history(self):
        if len(self._history) == 0:
            self._history.append(_history())
        if self._history_updated or self._working_history is None:
            self._working_history = self[:]

    def __getitem__(self, key):
        if len(self._history) == 0:
            self._history.append(_history())

        if isinstance(key, slice) or isinstance(key, int):
            h = self._history[key]
            if hasattr(h, '__len__'):
                if len(h) > 1: return h[0].merge(*h[1:])
                else: return h[0]
            return h
        elif isinstance(key, str):
            for i in self._history:
                if key == i.name:
                    return i
        elif type(key) in (tuple, list):
            h = [i for k in key for i in self._history if i.name == k]
            if len(h) > 1: return h[0].merge(*h[1:])
            else: return h[0]
        raise ValueError('Model index must be [slice],\
            [int] or [str], or list of string')

    def get_working_history(self):
        self._check_current_working_history()
        return self._working_history

    def new_frame(self, name=None, description=None):
        self._history.append(_history(name, description))
        self._history_updated = True

    def drop_frame(self):
        if len(self._history) < 2:
            self._history = []
        else:
            self._history = self._history[:-1]
        self._working_history = None
        self._history_updated = True

    def record(self, values, *tags):
        ''' Always write to the newest frame '''
        if len(self._history) == 0:
            self._history.append(_history())
        self._history[-1].record(values, *tags)
        self._history_updated = True

    def update(self, tags, new, after=None, before=None, n=None, absolute=False):
        if len(self._history) == 0:
            self._history.append(_history())
        self._history[-1].update(self, tags, new,
            after, before, n, absolute)
        self._history_updated = True

    def select(self, tags, default=None, after=None, before=None, n=None,
        filter_value=None, absolute=False, newest=False, return_time=False):
        ''' Query in history, default working history is the newest frame.

        Parameters
        ----------
        tags : list, str, filter function or any comparable object
            get all values contain given tags
        default : object
            default return value of found nothing
        after, before : time constraint (in millisecond)
            after < t < before
        n : int
            number of record return
        filter_value : function(value)
            function to filter each value found
        absolute : boolean
            whether required the same set of tags or just contain
        newest : boolean
            returning order (newest first, default is False)
        time : boolean
            whether return time tags

        Returns
        -------
        return : list
            always return list, in case of no value, return empty list
        '''
        self._check_current_working_history()
        return self._working_history.select(tags, default, after, before, n,
            filter_value, absolute, newest, return_time)

    def print_history(self):
        self._check_current_working_history()
        self._working_history.print_history()

    def print_frames(self):
        if len(self._history) == 0:
            self._history.append(_history())

        for i in self._history:
            logger.log(i)

    def print_code(self):
        import inspect
        if self._model_func is not None:
            logger.log(inspect.getsource(self._model_func))

    def __str__(self):
        s = ''
        s += 'Model: %s' % self._save_path + '\n'
        # weight
        s += '======== Weights ========\n'
        nb_params = 0
        # it is critical to get_weights here to fetch weight from created model
        for w in self.get_weights():
            s += ' - shape:%s' % str(w.shape) + '\n'
            nb_params += np.prod(w.shape)
        s += ' => Total: %d (parameters)' % nb_params + '\n'
        s += ' => Size: %.2f MB' % (nb_params * 4. / 1024. / 1024.) + '\n'
        # history
        self._check_current_working_history()
        s += str(self._working_history)

        # model function
        s += '======== Code ========\n'
        s += ' - api:%s' % self._api + '\n'
        s += ' - name:%s' % self._model_name + '\n'
        s += ' - args:%s' % str(self._model_args) + '\n'
        s += ' - sandbox:%s' % str(self._sandbox) + '\n'
        return s[:-1]

    # ==================== Load & Save ==================== #
    def save(self, path=None):
        '''
        '''
        if path is None and self._save_path is None:
            raise ValueError("Save path haven't specified!")
        path = path if path is not None else self._save_path
        self._save_path = path

        import cPickle
        import marshal
        from array import array

        f = h5py.File(path, 'w')
        f['history'] = cPickle.dumps(self._history)

        # ====== Save model function ====== #
        if self._model_func is not None:
            model_func = marshal.dumps(self._model_func.func_code)
            b = array("B", model_func)
            f['model_func'] = cPickle.dumps(b)
            f['model_args'] = cPickle.dumps(self._model_args)
            f['model_name'] = self._model_name
            f['sandbox'] = cPickle.dumps(self._sandbox)
            f['api'] = self._api

        # check weights, always fetch newest weights from model
        weights = self.get_weights()
        for i, w in enumerate(weights):
            f['weight_%d' % i] = w
        f['nb_weights'] = len(weights)

        #end
        f.close()

    @staticmethod
    def load(path):
        ''' Load won't create any modification to original AI file '''
        if not os.path.exists(path):
            m = model()
            m._save_path = path
            return m
        import cPickle
        import marshal

        m = model()
        m._save_path = path

        f = h5py.File(path, 'r')

        # ====== Load history ====== #
        if 'history' in f:
            m._history = cPickle.loads(f['history'].value)
        else:
            m._history = []

        # ====== Load model ====== #
        if 'api' in f:
            m._api = f['api'].value
        else: m._api = None
        if 'model_name' in f:
            m._model_name = f['model_name'].value
        else: m._model_name = None
        if 'model_args' in f:
            m._model_args = cPickle.loads(f['model_args'].value)
        else: m._model_args = None
        if 'sandbox' in f:
            m._sandbox = cPickle.loads(f['sandbox'].value)
        else: m._sandbox = {}

        # load model_func code
        if 'model_func' in f:
            b = cPickle.loads(f['model_func'].value)
            m._model_func = marshal.loads(b.tostring())
            sandbox = globals().copy() # create sandbox
            sandbox.update(deserialize_sandbox(m._sandbox))
            m._model_func = types.FunctionType(
                                m._model_func, sandbox, m._model_name)
        else: m._model_func = None

        # load weighs
        if 'nb_weights' in f:
            for i in xrange(f['nb_weights'].value):
                m._weights.append(f['weight_%d' % i].value)

        f.close()
        return m

# ======================================================================
# Trainer
# ======================================================================
class _iterator_wrapper(object):
    '''Fake class with iter function like dnntoolkit.batch'''

    def __init__(self, creator):
        super(_iterator_wrapper, self).__init__()
        self.creator = creator

    def iter(self, batch, shuffle, seed, mode, *args, **kwargs):
        ''' Create and return an iterator'''
        creator = self.creator
        if hasattr(creator, '__call__'):
            return creator(batch, shuffle, seed)
        elif hasattr(creator, 'next'):
            creator, news = itertools.tee(creator)
            self.creator = creator
            return news
        else:
            raise ValueError(
                'Creator of data for trainer must be a function or iterator')

def _callback(trainer):
    pass

def _parse_data_config(task, data):
    '''return train,valid,test'''
    train = None
    test = None
    valid = None
    if type(data) in (tuple, list):
        # only specified train data
        if type(data[0]) not in (tuple, list):
            if 'train' in task: train = data
            elif 'test' in task: test = data
            elif 'valid' in task: valid = data
        else: # also specified train and valid
            if len(data) == 1:
                if 'train' in task: train = data[0]
                elif 'test' in task: test = data[0]
                elif 'valid' in task: valid = data[0]
            if len(data) == 2:
                if 'train' in task: train = data[0]; valid = data[1]
                elif 'test' in task: test = data[0]
                elif 'valid' in task: valid = data[0]
            elif len(data) == 3:
                train = data[0]
                test = data[1]
                valid = data[2]
    elif type(data) == dict:
        if 'train' in data: train = data['train']
        if 'test' in data: test = data['test']
        if 'valid' in data: valid = data['valid']
    elif data is not None:
        if 'train' in task: train = [data]
        if 'test' in task: test = [data]
        if 'valid' in task: valid = [data]
    return train, valid, test

class trainer(object):

    """
    TODO:
     - custom data (not instance of dataset),
     - cross training 2 dataset,
     - custom action trigger under certain condition
     - layers configuration: ['dropout':0.5, 'noise':'0.075']
     - default ArgumentsParser
     - Add iter_mode, start, end to set_strategy
     - Add prediction task
    Value can be queried on callback:
     - idx(int): current run idx in the strategies, start from 0
     - cost: current training, testing, validating cost
     - iter(int): number of iteration, start from 0
     - data: current data (batch_start)
     - epoch(int): current epoch, start from 0
     - task(str): current task 'train', 'test', 'valid'
    Command can be triggered when running:
     - stop()
     - valid()
     - restart()
    """

    def __init__(self):
        super(trainer, self).__init__()
        self._seed = RandomState(MAGIC_SEED)
        self._strategy = []

        self._train_data = None
        self._valid_data = None
        self._test_data = None

        self.idx = 0 # index in strategy
        self.cost = None
        self.iter = None
        self.data = None
        self.epoch = 0
        self.task = None

        # callback
        self._epoch_start = _callback
        self._epoch_end = _callback
        self._batch_start = _callback
        self._batch_end = _callback
        self._train_start = _callback
        self._train_end = _callback
        self._valid_start = _callback
        self._valid_end = _callback
        self._test_start = _callback
        self._test_end = _callback

        self._stop = False
        self._valid_now = False
        self._restart_now = False

        self._log_enable = True
        self._log_newline = False

        self._cross_data = None
        self._pcross = 0.3

        self._iter_mode = 1

    # ==================== Trigger Command ==================== #
    def stop(self):
        ''' Stop current activity of this trainer immediatelly '''
        self._stop = True

    def valid(self):
        ''' Trigger validation immediatelly, asap '''
        self._valid_now = True

    def restart(self):
        ''' Trigger restart current process immediatelly '''
        self._restart_now = True

    # ==================== Setter ==================== #
    def set_action(self, name, action,
                   epoch_start=False, epoch_end=False,
                   batch_start=False, batch_end=False,
                   train_start=False, train_end=False,
                   valid_start=False, valid_end=False,
                   test_start=False, test_end=False):
        pass

    def set_log(self, enable=True, newline=False):
        self._log_enable = enable
        self._log_newline = newline

    def set_iter_mode(self, mode):
        '''
        ONly for training, for validation and testing mode = 0
        mode : 0, 1, 2
            0 - default, sequentially read each dataset
            1 - parallel read: proportionately for each dataset (e.g. batch_size=512,
                dataset1_size=1000, dataset2_size=500 => ds1=341, ds2=170)
            2 - parallel read: make all dataset equal size by over-sampling
                smaller dataset (e.g. batch_size=512, there are 5 dataset
                => each dataset 102 samples) (only work if batch size <<
                dataset size)
            3 - parallel read: make all dataset equal size by under-sampling
                smaller dataset (e.g. batch_size=512, there are 5 dataset
                => each dataset 102 samples) (only work if batch size <<
                dataset size)        '''
        self._iter_mode = mode

    def set_dataset(self, data, train=None, valid=None,
        test=None, cross=None, pcross=None):
        ''' Set dataset for trainer.

        Parameters
        ----------
        data : dnntoolkit.dataset
            dataset instance which contain all your data
        train : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of dataset used for training
        valid : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of dataset used for validation
        test : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of dataset used for testing
        cross : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of dataset used for cross training
        pcross : float (0.0-1.0)
            probablity of doing cross training when training, None=default=0.3

        Returns
        -------
        return : trainer
            for chaining method calling

        Note
        ----
        the order of train, valid, test must be the same in model function
        any None input will be ignored
        '''
        if isinstance(data, str):
            data = dataset(data, mode='r')
        if not isinstance(data, dataset):
            raise ValueError('[data] must be instance of dataset')

        self._dataset = data

        if train is not None:
            if type(train) not in (tuple, list):
                train = [train]
            self._train_data = train

        if valid is not None:
            if type(valid) not in (tuple, list):
                valid = [valid]
            self._valid_data = valid

        if test is not None:
            if type(test) not in (tuple, list):
                test = [test]
            self._test_data = test

        if cross is not None:
            if type(cross) not in (tuple, list):
                cross = [cross]
            self._cross_data = cross
        if self._pcross:
            self._pcross = pcross
        return self

    def set_model(self, cost_func=None, updates_func=None):
        ''' Set main function for this trainer to manipulate your model.

        Parameters
        ----------
        cost_func : theano.Function, function
            cost function: inputs=[X,y]
                           return: cost
        updates_func : theano.Function, function
            updates parameters function: inputs=[X,y]
                                         updates: parameters
                                         return: cost while training

        Returns
        -------
        return : trainer
            for chaining method calling
        '''
        if cost_func is not None and not hasattr(cost_func, '__call__'):
           raise ValueError('cost_func must be function')
        if updates_func is not None and not hasattr(updates_func, '__call__'):
           raise ValueError('updates_func must be function')

        self._cost_func = cost_func
        self._updates_func = updates_func
        return self

    def set_callback(self, epoch_start=_callback, epoch_end=_callback,
                     batch_start=_callback, batch_end=_callback,
                     train_start=_callback, train_end=_callback,
                     valid_start=_callback, valid_end=_callback,
                     test_start=_callback, test_end=_callback):
        ''' Set Callback while training, validating or testing the model.

        Parameters
        ----------
            all arguments is in form of:
                def function(trainer): ...
        Returns
        -------
        return : trainer
            for chaining method calling
        '''
        self._epoch_start = epoch_start
        self._epoch_end = epoch_end
        self._batch_start = batch_start
        self._batch_end = batch_end

        self._train_start = train_start
        self._valid_start = valid_start
        self._test_start = test_start

        self._train_end = train_end
        self._valid_end = valid_end
        self._test_end = test_end
        return self

    def set_strategy(self, task=None, data=None,
                     epoch=1, batch=512, validfreq=0.4,
                     shuffle=True, seed=None, yaml=None,
                     cross=None, pcross=None):
        ''' Set strategy for training.

        Parameters
        ----------
        task : str
            train, valid, or test
        data : str, list(str), map
            for training, data contain all training data and validation data names
            example: [['X_train','y_train'],['X_valid','y_valid']]
                  or {'train':['X_train','y_train'],'valid':['X_valid','y_valid']}
            In case of missing data, strategy will take default data names form
            set_dataset method
        epoch : int
            number of epoch for training (NO need for valid and test)
        batch : int, 'auto'
            number of samples for each batch
        validfreq : int(number of iteration), float(0.-1.)
            validation frequency when training, when float, it mean percentage
            of dataset
        shuffle : boolean
            shuffle dataset while training
        seed : int
            set seed for shuffle so the result is reproducible
        yaml : str
            path to yaml strategy file. When specify this arguments,
            all other arguments are ignored
        cross : str, list(str), numpy.ndarray
            list of dataset used for cross training
        pcross : float (0.0-1.0)
            probablity of doing cross training when training

        Returns
        -------
        return : trainer
            for chaining method calling
        '''
        if yaml is not None:
            import yaml as yaml_
            f = open(yaml, 'r')
            strategy = yaml_.load(f)
            f.close()
            for s in strategy:
                if 'dataset' in s:
                    self._dataset = dataset(s['dataset'])
                    continue
                if 'validfreq' not in s: s['validfreq'] = validfreq
                if 'batch' not in s: s['batch'] = batch
                if 'epoch' not in s: s['epoch'] = epoch
                if 'shuffle' not in s: s['shuffle'] = shuffle
                if 'data' not in s: s['data'] = data
                if 'cross' not in s: s['cross'] = cross
                if 'pcross' not in s: s['pcross'] = pcross
                if 'seed' in s: self._seed = RandomState(seed)
                self._strategy.append(s)
            return

        if task is None:
            raise ValueError('Must specify both [task] and [data] arguments')

        self._strategy.append({
            'task': task,
            'data': data,
            'epoch': epoch,
            'batch': batch,
            'shuffle': shuffle,
            'validfreq': validfreq,
            'cross': cross,
            'pcross': pcross
        })
        if seed is not None:
            self._seed = RandomState(seed)
        return self

    # ==================== Helper function ==================== #
    def _early_stop(self):
        # just a function reset stop flag and return its value
        tmp = self._stop
        self._stop = False
        return tmp

    def _early_valid(self):
        # just a function reset valid flag and return its value
        tmp = self._valid_now
        self._valid_now = False
        return tmp

    def _early_restart(self):
        # just a function reset valid flag and return its value
        tmp = self._restart_now
        self._restart_now = False
        return tmp

    def _get_str_datalist(self, datalist):
        if not datalist:
            return 'None'
        return ', '.join(['<Array: ' + str(i.shape) + '>'
                          if isinstance(i, np.ndarray) else str(i)
                          for i in datalist])

    def _check_dataset(self, data):
        ''' this function convert all pair of:
        [dataset, dataset_name] -> [batch_object]
        '''
        batches = []
        for i in data:
            if (type(i) in (tuple, list) and isinstance(i[0], np.ndarray)) or \
                isinstance(i, np.ndarray):
                batches.append(batch(arrays=i))
            elif isinstance(i, batch):
                batches.append(i)
            elif hasattr(i, '__call__') or hasattr(i, 'next'):
                batches.append(_iterator_wrapper(i))
            else:
                batches.append(self._dataset[i])
        return batches

    def _create_iter(self, data, batch, shuffle, mode, cross=None, pcross=0.3):
        ''' data: is [dnntoolkit.batch] instance, always a list
            cross: is [dnntoolkit.batch] instance, always a list
        '''
        seed = self._seed.randint(0, 10e8)
        data = [i.iter(batch, shuffle=shuffle, seed=seed, mode=mode)
                for i in data]
        # handle case that 1 batch return all data
        if len(data) == 1:
            iter_data = data[0]
        else:
            iter_data = zip(*data)
        if cross: # enable cross training
            seed = self._seed.randint(0, 10e8)
            if len(cross) == 1: # create cross iteration
                cross_it = cross[0].iter(batch, shuffle=shuffle,
                                  seed=seed, mode=mode)
            else:
                cross_it = zip(*[i.iter(batch, shuffle=shuffle,
                                    seed=seed, mode=mode)
                                 for i in cross])
            for d in iter_data:
                if random.random() < pcross:
                    try:
                        yield cross_it.next()
                    except StopIteration:
                        seed = self._seed.randint(0, 10e8)
                        if len(cross) == 1: # recreate cross iteration
                            cross_it = cross[0].iter(batch, shuffle=shuffle,
                                              seed=seed, mode=mode)
                        else:
                            cross_it = zip(*[i.iter(batch, shuffle=shuffle,
                                                seed=seed, mode=mode)
                                             for i in cross])
                yield d
        else: # only normal training
            for d in iter_data:
                yield d

    def _finish_train(self, train_cost, restart=False):
        self.cost = train_cost
        self._train_end(self) # callback
        self.cost = None
        self.task = None
        self.data = None
        self.it = 0
        return not restart

    # ==================== Main workflow ==================== #
    def _cost(self, task, valid_data, batch):
        '''
        Return
        ------
        True: finished the task
        False: restart the task
        '''
        self.task = task
        self.iter = 0
        if task == 'valid':
            self._valid_start(self)
        elif task == 'test':
            self._test_start(self)

        # convert name and ndarray to [dnntoolkit.batch] object
        valid_data = self._check_dataset(valid_data)
        # find n_samples
        n_samples = [len(i) for i in valid_data if hasattr(i, '__len__')]
        if len(n_samples) == 0: n_samples = 10e8
        else: n_samples = max(n_samples)
        # init some things
        valid_cost = []
        n = 0
        it = 0
        for data in self._create_iter(valid_data, batch, False, 0):
            # batch start
            it += 1
            n += data[0].shape[0]
            self.data = data
            self.iter = it
            self._batch_start(self)

            # main cost
            cost = self._cost_func(*self.data)
            if len(cost.shape) == 0:
                valid_cost.append(cost.tolist())
            else:
                valid_cost += cost.tolist()

            if self._log_enable:
                logger.progress(n, max_val=n_samples,
                    title='%s:Cost:%.4f' % (task, np.mean(cost)),
                    newline=self._log_newline, idx=task)

            # batch end
            self.cost = cost
            self.iter = it
            self._batch_end(self)
            self.data = None
            self.cost = None
        # ====== statistic of validation ====== #
        self.cost = valid_cost
        logger.log('\n => %s Stats: Mean:%.4f Var:%.2f Med:%.2f Min:%.2f Max:%.2f' %
                (task, np.mean(self.cost), np.var(self.cost), np.median(self.cost),
                np.percentile(self.cost, 5), np.percentile(self.cost, 95)))
        # ====== callback ====== #
        if task == 'valid':
            self._valid_end(self) # callback
        else:
            self._test_end(self)
        # ====== reset all flag ====== #
        self.cost = None
        self.task = None
        self.iter = 0
        return True

    def _train(self, train_data, valid_data, epoch, batch, validfreq, shuffle,
               cross=None, pcross=0.3):
        '''
        Return
        ------
        True: finished the task
        False: restart the task
        '''
        self.task = 'train'
        self.iter = 0
        self._train_start(self)
        it = 0
        # convert name and ndarray to [dnntoolkit.batch] object
        train_data = self._check_dataset(train_data)
        if cross:
            cross = self._check_dataset(cross)
        # get n_samples in training
        ntrain = [i.iter_len(self._iter_mode) for i in train_data
                  if hasattr(i, 'iter_len')]
        if len(ntrain) == 0: ntrain = 20 * batch
        else: ntrain = ntrain[0]
        # validfreq_it: number of iterations after each validation
        validfreq_it = 1
        if validfreq > 1.0:
            validfreq_it = int(validfreq)
        # ====== start ====== #
        train_cost = []
        for i in xrange(epoch):
            self.epoch = i
            self.iter = it
            self._epoch_start(self) # callback
            if self._early_stop(): # earlystop
                return self._finish_train(train_cost, self._early_restart())
            epoch_cost = []
            n = 0
            # ====== start batches ====== #
            for data in self._create_iter(train_data, batch, shuffle,
                                          self._iter_mode,
                                          cross, pcross):
                # start batch
                n += data[0].shape[0]
                # update ntrain constantly, if iter_mode = 1, no idea how many
                # data point in the dataset because of upsampling
                ntrain = max(ntrain, n)
                self.data = data
                self.iter = it
                self._batch_start(self) # callback
                if self._early_stop(): # earlystop
                    return self._finish_train(train_cost, self._early_restart())

                # main updates
                cost = self._updates_func(*self.data)

                # log
                epoch_cost.append(cost)
                train_cost.append(cost)
                if self._log_enable:
                    logger.progress(n, max_val=ntrain,
                        title='Epoch:%d,Iter:%d,Cost:%.4f' % (i + 1, it, cost),
                        newline=self._log_newline, idx='train')

                # end batch
                self.cost = cost
                self.iter = it
                self._batch_end(self)  # callback
                self.data = None
                self.cost = None
                if self._early_stop(): # earlystop
                    return self._finish_train(train_cost, self._early_restart())

                # validation, must update validfreq_it because ntrain updated also
                if validfreq <= 1.0:
                    validfreq_it = int(max(validfreq * ntrain / batch, 1))
                it += 1 # finish 1 iteration
                if (it % validfreq_it == 0) or self._early_valid():
                    if valid_data is not None:
                        self._cost('valid', valid_data, batch)
                        if self._early_stop(): # earlystop
                            return self._finish_train(train_cost, self._early_restart())
                    self.task = 'train' # restart flag back to train
            # ====== end epoch: statistic of epoch cost ====== #
            self.cost = epoch_cost
            self.iter = it
            logger.log('\n => Epoch Stats: Mean:%.4f Var:%.2f Med:%.2f Min:%.2f Max:%.2f' %
                    (np.mean(self.cost), np.var(self.cost), np.median(self.cost),
                    np.percentile(self.cost, 5), np.percentile(self.cost, 95)))

            self._epoch_end(self) # callback
            self.cost = None
            if self._early_stop(): # earlystop
                return self._finish_train(train_cost, self._early_restart())

        # end training
        return self._finish_train(train_cost, self._early_restart())

    def debug(self):
        raise NotImplementedError()

    def run(self):
        ''' run specified strategies
        Returns
        -------
        return : bool
            if exception raised, return False, otherwise return True
        '''
        try:
            while self.idx < len(self._strategy):
                config = self._strategy[self.idx]
                task = config['task']
                train, valid, test = _parse_data_config(task, config['data'])
                if train is None: train = self._train_data
                if test is None: test = self._test_data
                if valid is None: valid = self._valid_data
                cross = config['cross']
                pcross = config['pcross']
                if pcross is None: pcross = self._pcross
                if cross is None: cross = self._cross_data
                elif not hasattr(cross, '__len__'):
                    cross = [cross]

                epoch = config['epoch']
                batch = config['batch']
                validfreq = config['validfreq']
                shuffle = config['shuffle']

                if self._log_enable:
                    logger.log('\n******* %d-th run, with configuration: *******' % self.idx)
                    logger.log(' - Task:%s' % task)
                    logger.log(' - Train data:%s' % self._get_str_datalist(train))
                    logger.log(' - Valid data:%s' % self._get_str_datalist(valid))
                    logger.log(' - Test data:%s' % self._get_str_datalist(test))
                    logger.log(' - Cross data:%s' % self._get_str_datalist(cross))
                    logger.log(' - Cross prob:%s' % str(pcross))
                    logger.log(' - Epoch:%d' % epoch)
                    logger.log(' - Batch:%d' % batch)
                    logger.log(' - Validfreq:%d' % validfreq)
                    logger.log(' - Shuffle:%s' % str(shuffle))
                    logger.log('**********************************************')

                if 'train' in task:
                    if train is None:
                        logger.warning('*** no TRAIN data found, ignored **')
                    else:
                        while (not self._train(
                                train, valid, epoch, batch, validfreq, shuffle,
                                cross, pcross)):
                            pass
                elif 'valid' in task:
                    if valid is None:
                        logger.warning('*** no VALID data found, ignored **')
                    else:
                        while (not self._cost('valid', valid, batch)):
                            pass
                elif 'test' in task:
                    if test is None:
                        logger.warning('*** no TEST data found, ignored **')
                    else:
                        while (not self._cost('test', test, batch)):
                            pass
                # only increase idx after finish the task
                self.idx += 1
        except Exception, e:
            logger.error(str(e))
            import traceback; traceback.print_exc();
            return False
        return True

    # ==================== Debug ==================== #
    def __str__(self):
        s = '\n'
        s += 'Dataset:' + str(self._dataset) + '\n'
        s += 'Current run:%d' % self.idx + '\n'
        s += '============ \n'
        s += 'defTrain:' + self._get_str_datalist(self._train_data) + '\n'
        s += 'defValid:' + self._get_str_datalist(self._valid_data) + '\n'
        s += 'defTest:' + self._get_str_datalist(self._test_data) + '\n'
        s += 'defCross:' + self._get_str_datalist(self._cross_data) + '\n'
        s += 'pCross:' + str(self._pcross) + '\n'
        s += '============ \n'
        s += 'Cost_func:' + str(self._cost_func) + '\n'
        s += 'Updates_func:' + str(self._updates_func) + '\n'
        s += '============ \n'
        s += 'Epoch start:' + str(self._epoch_start) + '\n'
        s += 'Epoch end:' + str(self._epoch_end) + '\n'
        s += 'Batch start:' + str(self._batch_start) + '\n'
        s += 'Batch end:' + str(self._batch_end) + '\n'
        s += 'Train start:' + str(self._train_start) + '\n'
        s += 'Train end:' + str(self._train_end) + '\n'
        s += 'Valid start:' + str(self._valid_start) + '\n'
        s += 'Valid end:' + str(self._valid_end) + '\n'
        s += 'Test start:' + str(self._test_start) + '\n'
        s += 'Test end:' + str(self._test_end) + '\n'

        for i, st in enumerate(self._strategy):
            train, valid, test = _parse_data_config(st['task'], st['data'])
            if train is None: train = self._train_data
            if test is None: test = self._test_data
            if valid is None: valid = self._valid_data
            cross = st['cross']
            pcross = st['pcross']
            if cross and not hasattr(cross, '__len__'):
                cross = [cross]

            s += '====== Strategy %d-th ======\n' % i
            s += ' - Task:%s' % st['task'] + '\n'
            s += ' - Train:%s' % self._get_str_datalist(train) + '\n'
            s += ' - Valid:%s' % self._get_str_datalist(valid) + '\n'
            s += ' - Test:%s' % self._get_str_datalist(test) + '\n'
            s += ' - Cross:%s' % self._get_str_datalist(cross) + '\n'
            s += ' - pCross:%s' % str(pcross) + '\n'
            s += ' - Epoch:%d' % st['epoch'] + '\n'
            s += ' - Batch:%d' % st['batch'] + '\n'
            s += ' - Shuffle:%s' % st['shuffle'] + '\n'

        return s

# ======================================================================
# Data Preprocessing
# ======================================================================
def create_batch(n_samples, batch_size,
    start=None, end=None, prng=None, upsample=None, keep_size=False):
    '''
    No gaurantee that this methods will return the extract batch_size

    Parameters
    ----------
    n_samples : int
        size of original full dataset (not count start and end)
    prng : numpy.random.RandomState
        if prng != None, the upsampling process will be randomized
    upsample : int
        upsample > n_samples, batch will be sampled from original data to make
        the same total number of sample
        if [start] and [end] are specified, upsample will be rescaled according
        to original n_samples

    Example
    -------
    >>> from numpy.random import RandomState
    >>> create_batch(100, 17, start=0.0, end=1.0)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    >>> create_batch(100, 17, start=0.0, end=1.0, upsample=130)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (0, 20), (20, 37)]
    >>> create_batch(100, 17, start=0.0, end=1.0, prng=RandomState(12082518), upsample=130)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (20, 40), (80, 90)]

    Notes
    -----
    If you want to generate similar batch everytime, set the same seed before
    call this methods
    For odd number of batch and block, a goal of Maximize number of n_block and
    n_batch are applied
    '''
    #####################################
    # 1. Validate arguments.
    if start is None or start >= n_samples or start < 0:
        start = 0
    if end is None or end > n_samples:
        end = n_samples
    if end < start: #swap
        tmp = start
        start = end
        end = tmp

    if start < 1.0:
        start = int(start * n_samples)
    if end <= 1.0:
        end = int(end * n_samples)
    orig_n_samples = n_samples
    n_samples = end - start

    if upsample is None:
        upsample = n_samples
    else: # rescale
        upsample = int(upsample * float(n_samples) / orig_n_samples)
    #####################################
    # 2. Init.
    jobs = []
    n_batch = float(n_samples / batch_size)
    if n_batch < 1 and keep_size:
        raise ValueError('Cannot keep size when number of data < batch size')
    i = -1
    for i in xrange(int(n_batch)):
        jobs.append((start + i*batch_size, start + (i + 1) * batch_size))
    if not n_batch.is_integer():
        if keep_size:
            jobs.append((end - batch_size, end))
        else:
            jobs.append((start + (i + 1) * batch_size, end))

    #####################################
    # 3. Upsample jobs.
    upsample_mode = True if upsample >= n_samples else False
    upsample_jobs = []
    n = n_samples if upsample_mode else 0
    i = 0
    while n < upsample:
        # pick a package
        if prng is None:
            added_job = jobs[i % len(jobs)]
            i += 1
        elif prng is not None:
            added_job = jobs[prng.randint(0, len(jobs))]
        tmp = added_job[1] - added_job[0]
        if not keep_size: # only remove redundant size if not keep_size
            if n + tmp > upsample:
                tmp = n + tmp - upsample
                added_job = (added_job[0], added_job[1] - tmp)
        n += added_job[1] - added_job[0]
        # done
        upsample_jobs.append(added_job)

    if upsample_mode:
        return jobs + upsample_jobs
    else:
        return upsample_jobs

def _auto_batch_size(shape):
    # This method calculate based on reference to imagenet size
    batch = 256
    ratio = np.prod(shape[1:]) / (224 * 224 * 3)
    batch /= ratio
    return 2**int(np.log2(batch))

def _hdf5_get_all_dataset(hdf, fileter_func=None, path='/'):
    res = []
    # init queue
    q = queue()
    for i in hdf[path].keys():
        q.put(i)
    # get list of all file
    while not q.empty():
        p = q.pop()
        if 'Dataset' in str(type(hdf[p])):
            if fileter_func is not None and not fileter_func(p):
                continue
            res.append(p)
        elif 'Group' in str(type(hdf[p])):
            for i in hdf[p].keys():
                q.put(p + '/' + i)
    return res

def _hdf5_append_to_dataset(hdf_dataset, data):
    curr_size = hdf_dataset.shape[0]
    hdf_dataset.resize(curr_size + data.shape[0], 0)
    hdf_dataset[curr_size:] = data


class _dummy_shuffle():

    @staticmethod
    def shuffle(x):
        pass

    @staticmethod
    def permutation(x):
        return np.arange(x)

class batch(object):

    """Batch object
    Parameters
    ----------
    key : str, list(str)
        list of dataset key in h5py file
    hdf : h5py.File
        a h5py.File or list of h5py.File
    arrays : numpy.ndarray, list(numpy.ndarray)
        if arrays is specified, ignore key and hdf

    """

    def __init__(self, key=None, hdf=None, arrays=None):
        super(batch, self).__init__()
        self._normalizer = lambda x: x
        if arrays is None:
            if type(key) not in (tuple, list):
                key = [key]
            if type(hdf) not in (tuple, list):
                hdf = [hdf]
            if len(key) != len(hdf):
                raise ValueError('[key] and [hdf] must be equal size')

            self._key = key
            self._hdf = hdf
            self._data = []

            for key, hdf in zip(self._key, self._hdf):
                if key in hdf:
                    self._data.append(hdf[key])

            if len(self._data) > 0 and len(self._data) != len(self._hdf):
                raise ValueError('Not all [hdf] file contain given [key]')
            self._is_array_mode = False
        else:
            if type(arrays) not in (tuple, list):
                arrays = [arrays]
            self._data = arrays
            self._is_array_mode = True

    def _check(self, shape, dtype):
        # if not exist create initial dataset
        if len(self._data) == 0:
            for key, hdf in zip(self._key, self._hdf):
                if key not in hdf:
                    hdf.create_dataset(key, dtype=dtype, chunks=True,
                        shape=(0,) + shape[1:], maxshape=(None, ) + shape[1:])
                self._data.append(hdf[key])

        # check shape match
        for d in self._data:
            if d.shape[1:] != shape[1:]:
                raise TypeError('Shapes not match ' + str(d.shape) + ' - ' + str(shape))

    # ==================== Properties ==================== #
    def _is_dataset_init(self):
        if len(self._data) == 0:
            raise RuntimeError('Dataset have not initialized yet!')

    @property
    def shape(self):
        self._is_dataset_init()
        s = sum([i.shape[0] for i in self._data])
        return (s,) + i.shape[1:]

    @property
    def dtype(self):
        self._is_dataset_init()
        return self._data[0].dtype

    @property
    def value(self):
        self._is_dataset_init()
        if self._is_array_mode:
            return np.concatenate([i for i in self._data], axis=0)
        return np.concatenate([i.value for i in self._data], axis=0)

    def set_normalizer(self, normalizer):
        '''
        Parameters
        ----------
        normalizer : callable
            a function(X)
        '''
        if normalizer is None:
            self._normalizer = lambda x: x
        else:
            self._normalizer = normalizer
        return self

    # ==================== Arithmetic ==================== #
    def sum2(self, axis=0):
        ''' sum(X^2) '''
        self._is_dataset_init()
        if self._is_array_mode:
            return np.power(self[:], 2).sum(axis)

        s = 0
        isInit = False
        for X in self.iter(shuffle=False):
            X = X.astype(np.float64)
            if axis == 0:
                # for more stable precision
                s += np.sum(np.power(X, 2), 0)
            else:
                if not isInit:
                    s = [np.sum(np.power(X, 2), axis)]
                    isInit = True
                else:
                    s.append(np.sum(np.power(X, 2), axis))
        if isinstance(s, list):
            s = np.concatenate(s, axis=0)
        return s

    def sum(self, axis=0):
        ''' sum(X) '''
        self._is_dataset_init()
        if self._is_array_mode:
            return self[:].sum(axis)

        s = 0
        isInit = False
        for X in self.iter(shuffle=False):
            X = X.astype(np.float64)
            if axis == 0:
                # for more stable precision
                s += np.sum(X, 0)
            else:
                if not isInit:
                    s = [np.sum(X, axis)]
                    isInit = True
                else:
                    s.append(np.sum(X, axis))
        if isinstance(s, list):
            s = np.concatenate(s, axis=0)
        return s

    def mean(self, axis=0):
        self._is_dataset_init()

        s = self.sum(axis)
        return s / self.shape[axis]

    def var(self, axis=0):
        self._is_dataset_init()
        if self._is_array_mode:
            return np.var(np.concatenate(self._data, 0), axis)

        v2 = 0
        v1 = 0
        isInit = False
        n = self.shape[axis]
        for X in self.iter(shuffle=False):
            X = X.astype(np.float64)
            if axis == 0:
                v2 += np.sum(np.power(X, 2), axis)
                v1 += np.sum(X, axis)
            else:
                if not isInit:
                    v2 = [np.sum(np.power(X, 2), axis)]
                    v1 = [np.sum(X, axis)]
                    isInit = True
                else:
                    v2.append(np.sum(np.power(X, 2), axis))
                    v1.append(np.sum(X, axis))
        if isinstance(v2, list):
            v2 = np.concatenate(v2, axis=0)
            v1 = np.concatenate(v1, axis=0)
        v = v2 - 1 / n * np.power(v1, 2)
        return v / n

    # ==================== manupilation ==================== #
    def append(self, other):
        if not isinstance(other, np.ndarray):
            raise TypeError('Append only support numpy ndarray')
        self._check(other.shape, other.dtype)
        if self._is_array_mode:
            self._data = [np.concatenate((i, other), 0) for i in self._data]
        else: # hdf5
            for d in self._data:
                _hdf5_append_to_dataset(d, other)
        return self

    def duplicate(self, other):
        self._is_dataset_init()
        if not isinstance(other, int):
            raise TypeError('Only duplicate by int factor')
        if len(self._data) == 0:
            raise TypeError("Data haven't initlized yet")

        if self._is_array_mode:
            self._data = [np.concatenate([i] * other, 0) for i in self._data]
        else: # hdf5
            for d in self._data:
                n = d.shape[0]
                batch_size = int(max(0.1 * n, 1))
                for i in xrange(other - 1):
                    copied = 0
                    while copied < n:
                        copy = d[copied: int(min(copied + batch_size, n))]
                        _hdf5_append_to_dataset(d, copy)
                        copied += batch_size
        return self

    def sample(self, size, seed=None, proportion=True):
        '''
        Parameters
        ----------
        proportion : bool
            will the portion of each dataset in the batch reserved the same
            as in original dataset
        '''
        if seed:
            np.random.seed(seed)

        all_size = [i.shape[0] for i in self._data]
        s = sum(all_size)
        if proportion:
            idx = [sorted(
                    np.random.permutation(i)[:round(size * i / s)].tolist()
                   )
                   for i in all_size]
        else:
            size = int(size / len(all_size))
            idx = [sorted(np.random.permutation(i)[:size].tolist())
                   for i in all_size]
        return np.concatenate([i[j] for i, j in zip(self._data, idx)], 0)

    def _iter_fast(self, ds, batch_size, start=None, end=None,
            shuffle=True, seed=None):
        # craete random seed
        prng1 = None
        prng2 = _dummy_shuffle
        if shuffle:
            if seed is None:
                seed = MAGIC_SEED
            prng1 = RandomState(seed)
            prng2 = RandomState(seed)

        batches = create_batch(ds.shape[0], batch_size, start, end, prng1)
        prng2.shuffle(batches)
        for i, j in batches:
            data = ds[i:j]
            yield self._normalizer(data[prng2.permutation(data.shape[0])])

    def _iter_slow(self, batch_size=128, start=None, end=None,
                   shuffle=True, seed=None, mode=0):
        # ====== Set random seed ====== #
        all_ds = self._data[:]
        prng1 = None
        prng2 = _dummy_shuffle
        if shuffle:
            if seed is None:
                seed = MAGIC_SEED
            prng1 = RandomState(seed)
            prng2 = RandomState(seed)

        all_size = [i.shape[0] for i in all_ds]
        n_dataset = len(all_ds)

        # ====== Calculate batch_size ====== #
        if mode == 1: # equal
            s = sum(all_size)
            all_batch_size = [int(round(batch_size * i / s)) for i in all_size]
            for i in xrange(len(all_batch_size)):
                if all_batch_size[i] == 0: all_batch_size[i] += 1
            if sum(all_batch_size) > batch_size: # 0.5% -> round up, too much
                for i in xrange(len(all_batch_size)):
                    if all_batch_size[i] > 1:
                        all_batch_size[i] -= 1
                        break
            all_upsample = [None] * len(all_size)
        elif mode == 2 or mode == 3: # upsampling and downsampling
            maxsize = int(max(all_size)) if mode == 2 else int(min(all_size))
            all_batch_size = [int(batch_size / n_dataset) for i in xrange(n_dataset)]
            for i in xrange(batch_size - sum(all_batch_size)): # not enough
                all_batch_size[i] += 1
            all_upsample = [maxsize for i in xrange(n_dataset)]
        else: # sequential
            all_batch_size = [batch_size]
            all_upsample = [None]
            all_size = [sum(all_size)]
        # ====== Create all block and batches ====== #
        # [ ((idx1, batch1), (idx2, batch2), ...), # batch 1
        #   ((idx1, batch1), (idx2, batch2), ...), # batch 2
        #   ... ]
        all_block_batch = []
        # contain [block_batches1, block_batches2, ...]
        tmp_block_batch = []
        for n, batchsize, upsample in zip(all_size, all_batch_size, all_upsample):
            tmp_block_batch.append(
                create_batch(n, batchsize, start, end, prng1, upsample))
        # ====== Distribute block and batches ====== #
        if mode == 1 or mode == 2 or mode == 3:
            for i in zip_longest(*tmp_block_batch):
                all_block_batch.append([(k, v) for k, v in enumerate(i) if v is not None])
        else:
            all_size = [i.shape[0] for i in all_ds]
            all_idx = []
            for i, j in enumerate(all_size):
                all_idx += [(i, k) for k in xrange(j)] # (ds_idx, index)
            all_idx = [all_idx[i[0]:i[1]] for i in tmp_block_batch[0]]
            # complex algorithm to connecting the batch with different dataset
            for i in all_idx:
                tmp = []
                idx = i[0][0] # i[0][0]: ds_index
                start = i[0][1] # i[0][1]: index
                end = start
                for j in i[1:]: # detect change in index
                    if idx != j[0]:
                        tmp.append((idx, (start, end + 1)))
                        idx = j[0]
                        start = j[1]
                    end = j[1]
                tmp.append((idx, (start, end + 1)))
                all_block_batch.append(tmp)
        prng2.shuffle(all_block_batch)
        # print if you want debug
        # for _ in all_block_batch:
        #     for i, j in _:
        #         print('ds:', i, '  batch:', j)
        #     print('===== End =====')
        # ====== return iteration ====== #
        for _ in all_block_batch: # each _ is a block
            batches = np.concatenate(
                [all_ds[i][j[0]:j[1]] for i, j in _], axis=0)
            batches = batches[prng2.permutation(batches.shape[0])]
            yield self._normalizer(batches)

    def iter(self, batch_size=128, start=None, end=None,
        shuffle=True, seed=None, normalizer=None, mode=0):
        ''' Create iteration for all dataset contained in this _batch
        When [start] and [end] are given, it mean appying for each dataset
        If the amount of data between [start] and [end] < 1.0

        Parameters
        ----------
        batch_size : int, 'auto'
            size of each batch (data will be loaded in big block 8 times
            larger than this size)
        start : int, float(0.0-1.0)
            start point in dataset, will apply for all dataset
        end : int, float(0.0-1.0)
            end point in dataset, will apply for all dataset
        shuffle : bool, str
            wheather enable shuffle
        seed : int
        normalizer : callable, function
            funciton will be applied to each batch before return
        mode : 0, 1, 2
            0 - default, sequentially read each dataset
            1 - parallel read: proportionately for each dataset (e.g.
                batch_size=512, dataset1_size=1000, dataset2_size=500
                => ds1=341, ds2=170)
            2 - parallel read (upsampling): upsampling smaller dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size <<
                dataset size)
            3 - parallel read (downsampling): downsampling larger dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size <<
                dataset size)

        Returns
        -------
        return : generator
            generator generate batches of data

        Notes
        -----
        This method is thread-safe, as it uses private instance of RandomState.
        To create consistent permutation of shuffled dataset, you must:
         - both batch have the same number and order of dataset
         - using the same seed and mode when calling iter()
        Hint: small level of batch shuffle can be obtained by using normalizer
        function
        The only case that 2 [dnntoolkit.batch] have the same order is when
        mode=0 and shuffle=False, for example
        >>> X = batch(['X1','X2], [f1, f2])
        >>> y = batch('Y', f)
        >>> X.iter(512, mode=0, shuffle=False) have the same order with
            y.iter(512, mode=0, shuffle=False)
        '''
        self._is_dataset_init()
        if normalizer is not None:
            self.set_normalizer(normalizer)
        if batch_size == 'auto':
            batch_size = _auto_batch_size(self.shape)

        if len(self._data) == 1:
            return self._iter_fast(self._data[0], batch_size, start, end,
                                   shuffle, seed)
        else:
            return self._iter_slow(batch_size, start, end, shuffle, seed,
                                   mode)

    def iter_len(self, mode):
        '''This methods return estimated iteration length'''
        self._is_dataset_init()
        if mode == 2: #upsampling
            maxlen = max([i.shape[0] for i in self._data])
            return int(maxlen * len(self._data))
        if mode == 3: #downsampling
            minlen = min([i.shape[0] for i in self._data])
            return int(minlen * len(self._data))
        return len(self)

    def __len__(self):
        ''' This method return actual len by sum all shape[0] '''
        self._is_dataset_init()
        return sum([i.shape[0] for i in self._data])

    def __getitem__(self, key):
        self._is_dataset_init()
        if type(key) == tuple:
            return np.concatenate(
                [d[k] for k, d in zip(key, self._data) if k is not None],
                axis=0)
        return np.concatenate([i[key] for i in self._data], axis=0)

    def __setitem__(self, key, value):
        '''
        '''
        self._is_dataset_init()
        self._check(value.shape, value.dtype)
        if isinstance(key, slice):
            for d in self._data:
                d[key] = value
        elif isinstance(key, tuple):
            for k, d in zip(key, self._data):
                if k is not None:
                    d[k] = value

    def __str__(self):
        if len(self._data) == 0:
            return '<batch: None>'
        s = '<batch: '
        if self._is_array_mode: key = [''] * len(self._data)
        else: key = self._key
        for k, d in zip(key, self._data):
            s += '[%s,%s,%s]-' % (k, d.shape, d.dtype)
        s = s[:-1] + '>'
        return s

class dataset(object):

    '''
    dataset object to manage multiple hdf5 file

    Note
    ----
    dataset['X1', 'X2']: will search for 'X1' in all given files, then 'X2',
    then 'X3' ...
    iter(batch_size, start, end, shuffle, seed, normalizer, mode)
        mode : 0, 1, 2
            0 - default, read one by one each dataset
            1 - equally read each dataset, upsampling smaller dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size << dataset size)
            2 - proportionately read each dataset (e.g. batch_size=512,
                dataset1_size=1000, dataset2_size=500 => ds1=341, ds2=170)
    '''

    def __init__(self, path, mode='r'):
        super(dataset, self).__init__()
        self._mode = mode
        if type(path) not in (list, tuple):
            path = [path]

        self._hdf = []
        # map: ("ds_name1","ds_name2",...): <_batch object>
        self._datamap = {}
        self._write_mode = None

        # all dataset have dtype('O') type will be return directly
        self._object = defaultdict(list)
        obj_type = np.dtype('O')
        # map: "ds_name":[hdf1, hdf2, ...]
        self._index = defaultdict(list)
        for p in path:
            f = h5py.File(p, mode=mode)
            self._hdf.append(f)
            ds = _hdf5_get_all_dataset(f)
            for i in ds:
                tmp = f[i]
                if tmp.dtype == obj_type or len(tmp.shape) == 0:
                    self._object[i].append(f)
                else:
                    self._index[i].append(f)

        self._isclose = False

    # ==================== Set and get ==================== #
    def set_write(self, mode):
        '''
        Parameters
        ----------
        mode : 'all', 'last', int(index), slice
            specify which hdf files will be used for write
        '''
        self._write_mode = mode

    def is_close(self):
        return self._isclose

    def get_all_dataset(self, fileter_func=None, path='/'):
        ''' Get all dataset contained in the hdf5 file.

        Parameters
        ----------
        filter_func : function
            filter function applied on the name of dataset to find
            appropriate dataset
        path : str
            path start searching for dataset

        Returns
        -------
        return : list(str)
            names of all dataset
        '''
        all_dataset = []
        for i in self._hdf:
            all_dataset += _hdf5_get_all_dataset(i, fileter_func, path)
        return all_dataset

    # ==================== Main ==================== #
    def _get_write_hdf(self):
        '''Alawys return list of hdf5 files'''
        # ====== Only 1 file ====== #
        if len(self._hdf) == 1:
            return self._hdf

        # ====== Multple files ====== #
        if self._write_mode is None:
            warnings.warn('Have not set write mode, default is [last]',
                          RuntimeWarning)
            self._write_mode = 'last'

        if self._write_mode == 'last':
            return [self._hdf[-1]]
        elif self._write_mode == 'all':
            return self._hdf
        elif isinstance(self._write_mode, str):
            return [h for h in self._hdf if h.filename == self._write_mode]
        elif type(self._write_mode) in (tuple, list):
            if isinstance(self._write_mode[0], int):
                return [self._hdf[i] for i in self._write_mode]
            else: # search all file with given name
                hdf = []
                for k in self._write_mode:
                    for h in self._hdf:
                        if h.filename == k:
                            hdf.append(h)
                            break
                return hdf
        else: # int or slice index
            hdf = self._hdf[self._write_mode]
            if type(hdf) not in (tuple, list):
                hdf = [hdf]
            return hdf

    def __getitem__(self, key):
        ''' Logic of this function:
         - Object is returned directly, even though key is mixture of object
         and array, only return objects
         - key is always processed as an array
         - Find as much as possible all hdf contain the key
         Example
         -------
         hdf1: a, b, c
         hdf2: a, b
         hdf3: a
         ==> key = a return [hdf1, hdf2, hdf3]
         ==> key = (a,b) return [hdf1, hdf2, hdf3, hdf1, hdf2]
        '''
        if type(key) not in (tuple, list):
            key = [key]

        # ====== Return object is priority ====== #
        ret = []
        for k in key:
            if k in self._object:
                ret += [i[k].value for i in self._object[k]]
        if len(ret) == 1:
            return ret[0]
        elif len(ret) > 0:
            return ret

        # ====== Return _batch ====== #
        keys = []
        for k in key:
            hdf = self._index[k]
            if len(hdf) == 0 and self._mode != 'r': # write mode activated
                hdf = self._get_write_hdf()
            ret += hdf
            keys += [k] * len(hdf)
        idx = tuple(keys + ret)

        if idx in self._datamap:
            return self._datamap[idx]
        b = batch(keys, ret)
        self._datamap[idx] = b
        return b

    def __setitem__(self, key, value):
        ''' Logic of this function:
         - If mode is 'r': NO write => error
         - if object, write the object directly
         - mode=all: write to all hdf
         - mode=last: write to the last one
         - mode=slice: write to selected
        '''
        # check input
        if self._mode == 'r':
            raise RuntimeError('No write is allowed in read mode')
        if type(key) not in (tuple, list):
            key = [key]

        # set str value directly
        if isinstance(value, str) or \
            (hasattr(value, 'dtype') and value.dtype == object):
            hdf = self._get_write_hdf()
            for i in hdf:
                for j in key:
                    i[j] = value
                    self._object[j].append(i)
        else: # array
            # find appropriate key
            hdf = self._get_write_hdf()
            for k in key: # each key
                for h in hdf: # do the same for all writable hdf
                    if k in h:
                        h[k][:] = value
                    else:
                        h[k] = value
                        self._index[k].append(h)

    def __contains__(self, key):
        r = False
        for hdf in self._hdf:
            if key in hdf:
                r += 1
        return r

    def close(self):
        try:
            for i in self._hdf:
                i.close()
        except:
            pass
        self._isclose = True

    def __del__(self):
        try:
            for i in self._hdf:
                i.close()
            del self._hdf
        except:
            pass
        self._isclose = True

    def __str__(self):
        s = 'Dataset contains: %d (files)' % len(self._hdf) + '\n'
        if self._isclose:
            s += '******** Closed ********\n'
        else:
            s += '******** Array ********\n'
            all_data = self._index.keys() # faster
            for i in all_data:
                all_hdf = self._index[i]
                for j in all_hdf:
                    s += ' - name:%-13s  shape:%-18s  dtype:%-8s  hdf:%s' % \
                        (i, j[i].shape, j[i].dtype, j.filename) + '\n'
            s += '******** Objects ********\n'
            all_data = self._object.keys() # faster
            for i in all_data:
                all_hdf = self._object[i]
                for j in all_hdf:
                    s += ' - name:%-13s  shape:%-18s  dtype:%-8s  hdf:%s' % \
                        (i, j[i].shape, j[i].dtype, j.filename) + '\n'
        return s[:-1]

    # ==================== Static loading ==================== #
    @staticmethod
    def load_mnist(path='https://s3.amazonaws.com/ai-datasets/mnist.hdf'):
        '''
        path : str
            local path or url to hdf5 datafile
        '''
        datapath = net.get_file('mnist.hdf', path)
        logger.info('Loading data from: %s' % datapath)
        return dataset(datapath, mode='r')

    def load_imdb(path):
        pass

    def load_reuters(path):
        pass

# ======================================================================
# Visualiztion
# ======================================================================
class visual():
    chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    @staticmethod
    def plot_confusion_matrix(cm, labels, axis=None, fontsize=13):
        from matplotlib import pyplot as plt

        title = 'Confusion matrix'
        cmap = plt.cm.Blues

        # column normalize
        if np.max(cm) > 1:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm_normalized = cm

        if axis is None:
            axis = plt.gca()

        im = axis.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        axis.set_title(title)
        axis.get_figure().colorbar(im)
        tick_marks = np.arange(len(labels))

        axis.set_xticks(tick_marks)
        axis.set_yticks(tick_marks)
        axis.set_xticklabels(labels, rotation=90, fontsize=13)
        axis.set_yticklabels(labels, fontsize=13)

        axis.set_ylabel('True label')
        axis.set_xlabel('Predicted label')
        # axis.tight_layout()
        return axis

    @staticmethod
    def plot_weights(x, ax=None, colormap = "Greys", colorbar=False, path=None, keep_aspect=True):
        '''
        Parameters
        ----------
        x : numpy.ndarray
            2D array
        ax : matplotlib.Axis
            create by fig.add_subplot, or plt.subplots
        colormap : str
            colormap alias from plt.cm.Greys = 'Greys'
        colorbar : bool, 'all'
            whether adding colorbar to plot, if colorbar='all', call this
            methods after you add all subplots will create big colorbar
            for all your plots
        path : str
            if path is specified, save png image to given path

        Notes
        -----
        Make sure nrow and ncol in add_subplot is int or this error will show up
         - ValueError: The truth value of an array with more than one element is
            ambiguous. Use a.any() or a.all()

        Example
        -------
        >>> x = np.random.rand(2000, 1000)
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(2, 2, 1)
        >>> dnntoolkit.visual.plot_weights(x, ax)
        >>> ax = fig.add_subplot(2, 2, 2)
        >>> dnntoolkit.visual.plot_weights(x, ax)
        >>> ax = fig.add_subplot(2, 2, 3)
        >>> dnntoolkit.visual.plot_weights(x, ax)
        >>> ax = fig.add_subplot(2, 2, 4)
        >>> dnntoolkit.visual.plot_weights(x, ax, path='/Users/trungnt13/tmp/shit.png')
        >>> plt.show()
        '''
        from matplotlib import pyplot as plt
        if colormap is None:
            colormap = plt.cm.Greys

        if x.ndim > 2:
            raise ValueError('No support for > 2D')
        elif x.ndim == 1:
            x = x[:, None]

        ax = ax if ax is not None else plt.gca()
        if keep_aspect:
            ax.set_aspect('equal', 'box')
        # ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_title(str(x.shape), fontsize=6)
        img = ax.pcolorfast(x, cmap=colormap, alpha=0.8)
        plt.grid(True)

        if colorbar == 'all':
            fig = ax.get_figure()
            axes = fig.get_axes()
            fig.colorbar(img, ax=axes)
        elif colorbar:
            plt.colorbar(img, ax=ax)

        if path:
            plt.savefig(path, dpi=300, format='png', bbox_inches='tight')
        return ax

    @staticmethod
    def plot_conv_weights(x, colormap = "Greys", path=None):
        '''
        Example
        -------
        >>> # 3D shape
        >>> x = np.random.rand(32, 28, 28)
        >>> dnntoolkit.visual.plot_conv_weights(x)
        >>> # 4D shape
        >>> x = np.random.rand(32, 3, 28, 28)
        >>> dnntoolkit.visual.plot_conv_weights(x)
        '''
        from matplotlib import pyplot as plt
        if colormap is None:
            colormap = plt.cm.Greys

        shape = x.shape
        if len(shape) == 3:
            ncols = int(np.ceil(np.sqrt(shape[0])))
            nrows = int(ncols)
        elif len(shape) == 4:
            ncols = shape[0]
            nrows = shape[1]
        else:
            raise ValueError('Unsupport for %d dimension' % x.ndim)

        fig = plt.figure()
        count = 0
        for i in xrange(nrows):
            for j in xrange(ncols):
                count += 1
                # skip
                if x.ndim == 3 and count > shape[0]:
                    continue

                ax = fig.add_subplot(nrows, ncols, count)
                ax.set_aspect('equal', 'box')
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0 and j == 0:
                    ax.set_xlabel('New channels', fontsize=6)
                    ax.xaxis.set_label_position('top')
                    ax.set_ylabel('Old channels', fontsize=6)
                    ax.yaxis.set_label_position('left')
                else:
                    ax.axis('off')
                # image data
                if x.ndim == 4:
                    img = ax.pcolorfast(x[j, i], cmap=colormap, alpha=0.8)
                else:
                    img = ax.pcolorfast(x[count - 1], cmap=colormap, alpha=0.8)
                plt.grid(True)

        # colorbar
        axes = fig.get_axes()
        fig.colorbar(img, ax=axes)

        if path:
            plt.savefig(path, dpi=300, format='png', bbox_inches='tight')
        return fig

    @staticmethod
    def print_hinton(arr, max_arr=None, return_str=False):
        ''' Print bar string, fast way to visual magnitude of value in terminal

        Example:
        -------
        >>> W = np.random.rand(10,10)
        >>> print_hinton(W)
        >>> ▁▃▄█▅█ ▅▃▅
        >>> ▅▂▆▄▄ ▅▅
        >>> ▄██▆▇▆▆█▆▅
        >>> ▄▄▅▂▂▆▅▁▅▆
        >>> ▂ ▁  ▁▄▆▅▁
        >>> ██▃█▃▃▆ ▆█
        >>>  ▁▂▁ ▁▃▃▆▂
        >>> ▅▂▂█ ▂ █▄▅
        >>> ▃▆▁▄▁▆▇▃▅▁
        >>> ▄▁▇ ██▅ ▂▃
        Returns
        -------
        return : str
            plot of array, for example: ▄▅▆▇
        '''
        arr = np.asarray(arr)
        if len(arr.shape) == 1:
            arr = arr[None, :]

        def visual_func(val, max_val):
            if abs(val) == max_val:
                step = len(visual.chars) - 1
            else:
                step = int(abs(float(val) / max_val) * len(visual.chars))
            colourstart = ""
            colourend = ""
            if val < 0:
                colourstart, colourend = '\033[90m', '\033[0m'
            return colourstart + visual.chars[step] + colourend

        if max_arr is None:
            max_arr = arr
        max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))
        # print(np.array2string(arr,
        #                       formatter={'float_kind': lambda x: visual(x, max_val)},
        #                       max_line_width=5000)
        # )
        f = np.vectorize(visual_func)
        result = f(arr, max_val) # array of ▄▅▆▇
        rval = ''
        for r in result:
            rval += ''.join(r) + '\n'
        if return_str:
            return rval[:-1]
        else:
            logger.log(rval)

    @staticmethod
    def print_bar(x, height=20.0, bincount=None, binwidth=None, pch="o",
                  title="", xlab=False, showSummary=False, regular=False):
        '''
        Parameters
        ----------
        x : list(number), numpy.ndarray, str(filepath)
            input array
        height : float
            the height of the histogram in # of lines
        bincount : int
            number of bins in the histogram
        binwidth : int
            width of bins in the histogram
        pch : str
            shape of the bars in the plot, e.g 'o'
        title : str
            title at the top of the plot, None = no title
        xlab : boolean
            whether or not to display x-axis labels
        showSummary : boolean
            whether or not to display a summary
        regular : boolean
            whether or not to start y-labels at 0

        '''
        try:
            import bashplotlib
            if hasattr(x, 'flatten'):
                x = x.flatten()
            s = bashplotlib.plot_bar(x, height=height, bincount=bincount,
                    binwidth=binwidth, pch=pch, colour="default",
                    title=title, xlab=xlab, showSummary=showSummary,
                    regular=regular, return_str=True)
            logger.log(s)
        except Exception, e:
            logger.warning('print_bar: Ignored! Error:%s' % str(e))

    @staticmethod
    def print_scatter(x, y, size=None, pch="o", title=""):
        '''
        Parameters
        ----------
        x : list, numpy.ndarray
            list of x series
        y : list, numpy.ndarray
            list of y series
        size : int
            width of plot
        pch : str
            any character to represent a points
        title : str
            title for the plot, None = not show
        '''
        try:
            import bashplotlib
            if hasattr(x, 'flatten'):
                x = x.flatten()
            if hasattr(y, 'flatten'):
                y = y.flatten()
            s = bashplotlib.plot_scatter(x, y, size=size, pch=pch,
                colour='default', title=title, return_str=True)
            logger.log(s)
        except Exception, e:
            logger.warning('print_scatter: Ignored! Error:%s' % str(e))

    @staticmethod
    def print_hist(x, height=20.0, bincount=None, binwidth=None, pch="o",
                  title="", xlab=False, showSummary=False, regular=False):
        '''
        Parameters
        ----------
        x : list(number), numpy.ndarray, str(filepath)
            input array
        height : float
            the height of the histogram in # of lines
        bincount : int
            number of bins in the histogram
        binwidth : int
            width of bins in the histogram
        pch : str
            shape of the bars in the plot, e.g 'o'
        title : str
            title at the top of the plot, None = no title
        xlab : boolean
            whether or not to display x-axis labels
        showSummary : boolean
            whether or not to display a summary
        regular : boolean
            whether or not to start y-labels at 0
        '''
        try:
            import bashplotlib
            if hasattr(x, 'flatten'):
                x = x.flatten()
            s = bashplotlib.plot_hist(x, height=height, bincount=bincount,
                    binwidth=binwidth, pch=pch, colour="default",
                    title=title, xlab=xlab, showSummary=showSummary,
                    regular=regular, return_str=True)
            logger.log(s)
        except Exception, e:
            logger.warning('print_hist: Ignored! Error:%s' % str(e))

    @staticmethod
    def plot_hinton(matrix, max_weight=None, ax=None):
        '''
        Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
        a weight matrix):
            Positive: white
            Negative: black
        squares, and the size of each square represents the magnitude of each value.
        * Note: performance significant decrease as array size > 50*50
        Example:
            W = np.random.rand(10,10)
            hinton_plot(W)
        '''
        from matplotlib import pyplot as plt

        """Draw Hinton diagram for visualizing a weight matrix."""
        ax = ax if ax is not None else plt.gca()

        if not max_weight:
            max_weight = 2**np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w))
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()
        return ax

    @staticmethod
    def plot_theano(var_or_func):
        theano.printing.pydotprint(var_or_func, outfile = '/Users/trungnt13/tmp/tmp.png')
        os.system('open -a /Applications/Preview.app /Users/trungnt13/tmp/tmp.png')

class logger():
    _last_value = 0
    _last_time = -1
    _default_logger = None
    _is_enable = True
    """docstring for Logger"""

    @staticmethod
    def set_enable(is_enable):
        logger._is_enable = is_enable

    @staticmethod
    def _check_init_logger():
        if logger._default_logger is None:
            logger.create_logger(logging_path=None)

    @staticmethod
    def set_print_level(level):
        ''' VERBOSITY level:
         - CRITICAL: 50
         - ERROR   : 40
         - WARNING : 30
         - INFO    : 20
         - DEBUG   : 10
         - UNSET   : 0
        '''
        logger._check_init_logger()
        logger._default_logger.handlers[0].setLevel(level)

    @staticmethod
    def set_save_path(logging_path, mode='w', multiprocess=False):
        '''All old path will be ignored'''
        import logging
        logger._check_init_logger()
        log = logger._default_logger
        log.handlers = [log.handlers[0]]

        if type(logging_path) not in (tuple, list):
            logging_path = [logging_path]

        for path in logging_path:
            if path is not None:
                # saving path
                fh = logging.FileHandler(path, mode=mode)
                fh.setFormatter(logging.Formatter(
                    fmt = '%(asctime)s %(levelname)s  %(message)s',
                    datefmt = '%d/%m/%Y %I:%M:%S'))
                fh.setLevel(logging.DEBUG)
                if multiprocess:
                    import multiprocessing_logging
                    log.addHandler(
                        multiprocessing_logging.MultiProcessingHandler('mpi', fh))
                else:
                    log.addHandler(fh)

    @staticmethod
    def warning(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        logger._default_logger.warning(*anything)

    @staticmethod
    def error(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        logger._default_logger.error(*anything)

    @staticmethod
    def critical(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        logger._default_logger.critical(*anything)

    @staticmethod
    def debug(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        logger._default_logger.debug(*anything)

    @staticmethod
    def info(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        if len(anything) == 0: logger._default_logger.info('')
        else: logger._default_logger.info(*anything)

    @staticmethod
    def log(*anything, **kwargs):
        '''This log is at INFO level'''
        if not logger._is_enable: return
        import logging
        logger._check_init_logger()
        # format with only messages
        for h in logger._default_logger.handlers:
            h.setFormatter(logging.Formatter(fmt = '%(message)s'))

        if len(anything) == 0:
            logger._default_logger.info('')
        else:
            logger._default_logger.info(*anything)

        # format with time and level
        for h in logger._default_logger.handlers:
            h.setFormatter(logging.Formatter(
                fmt = '%(asctime)s %(levelname)s  %(message)s',
                datefmt = '%d/%m/%Y %I:%M:%S'))

    _last_progress_idx = None

    @staticmethod
    def progress(p, max_val=1.0, title='Progress', bar='=', newline=False, idx=None):
        '''
        Parameters
        ----------
        p : number
            current progress value
        max_val : number
            maximum value progress can reach (be equal)
        idx : anything
            identification of current progress, if 2 progress is diffrent, print
            newline to switch to print new progress

        Notes
        -----
        This methods is not thread safe
        '''
        if not logger._is_enable:
            return
        # ====== Check same progress or not ====== #
        if logger._last_progress_idx != idx:
            print()
        logger._last_progress_idx = idx

        # ====== Config ====== #
        if p < 0: p = 0.0
        if p > max_val: p = max_val
        fmt_str = "\r%s (%.2f/%.2f)[%s] - ETA:%.2fs ETD:%.2fs"
        if max_val > 100:
            p = int(p)
            max_val = int(max_val)
            fmt_str = "\r%s (%d/%d)[%s] - ETA:%.2fs ETD:%.2fs"

        if newline:
            fmt_str = fmt_str[1:]
            fmt_str += '\n'
        # ====== ETA: estimated time of arrival ====== #
        if logger._last_time < 0:
            logger._last_time = time.time()
        eta = (max_val - p) / max(1e-13, abs(p - logger._last_value)) * (time.time() - logger._last_time)
        etd = time.time() - logger._last_time
        logger._last_value = p
        logger._last_time = time.time()
        # ====== print ====== #
        max_val_bar = 20
        n_bar = int(p / max_val * max_val_bar)
        bar = '=' * n_bar + '>' + ' ' * (max_val_bar - n_bar)
        sys.stdout.write(fmt_str % (title, p, max_val, bar, eta, etd))
        sys.stdout.flush()
        # if p >= max_val:
        #     sys.stdout.write("\n")

    @staticmethod
    def create_logger(name=None, logging_path=None, mode='w', multiprocess=False):
        ''' All logger are created at DEBUG level

        Parameters
        ----------

        Example
        -------
        >>> logger.debug('This is a debug message')
        >>> logger.info('This is an info message')
        >>> logger.warning('This is a warning message')
        >>> logger.error('This is an error message')
        >>> logger.critical('This is a critical error message')

        Note
        ----
        if name is None or default, the created logger will be used as default
        logger for dnntoolkit
        '''
        import logging
        if name is None:
            name = 'default'
        log = logging.getLogger('dnntoolkit.%s' % name)
        # remove all old handler
        log.handlers = []
        # print
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(
            fmt = '%(asctime)s %(levelname)s:  %(message)s',
            datefmt = '%d/%m/%Y %I:%M:%S'))
        sh.setLevel(logging.DEBUG)

        # add the current logger
        log.setLevel(logging.DEBUG)
        log.addHandler(sh)

        if type(logging_path) not in (tuple, list):
            logging_path = [logging_path]

        for path in logging_path:
            if path is not None:
                # saving path
                fh = logging.FileHandler(path, mode=mode)
                fh.setFormatter(logging.Formatter(
                    fmt = '%(asctime)s %(levelname)s  %(message)s',
                    datefmt = '%d/%m/%Y %I:%M:%S'))
                fh.setLevel(logging.DEBUG)
                if multiprocess:
                    import multiprocessing_logging
                    log.addHandler(
                        multiprocessing_logging.MultiProcessingHandler('mpi', fh))
                else:
                    log.addHandler(fh)

        # enable or disable
        if name == 'default':
            logger._default_logger = log
            logger.set_enable(logger._is_enable)
        return log

# ======================================================================
# Feature
# ======================================================================
class speech():

    """docstring for speech"""

    def __init__(self):
        super(speech, self).__init__()

    # ==================== Predefined datasets information ==================== #
    nist15_cluster_lang = OrderedDict([
        ['ara', ['ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb']],
        ['zho', ['zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu']],
        ['eng', ['eng-gbr', 'eng-usg', 'eng-sas']],
        ['fre', ['fre-waf', 'fre-hat']],
        ['qsl', ['qsl-pol', 'qsl-rus']],
        ['spa', ['spa-car', 'spa-eur', 'spa-lac', 'por-brz']]
    ])
    nist15_lang_list = np.asarray([
        # Egyptian, Iraqi, Levantine, Maghrebi, Modern Standard
        'ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb',
        # Cantonese, Mandarin, Min Dong, Wu
        'zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu',
        # British, American, South Asian (Indian)
        'eng-gbr', 'eng-usg', 'eng-sas',
        # West african, Haitian
        'fre-waf', 'fre-hat',
        # Polish, Russian
        'qsl-pol', 'qsl-rus',
        # Caribbean, European, Latin American, Brazilian
        'spa-car', 'spa-eur', 'spa-lac', 'por-brz'])

    @staticmethod
    def nist15_label(label):
        '''
        Return
        ------
        lang_id : int
            idx in the list of 20 language, None if not found
        cluster_id : int
            idx in the list of 6 clusters, None if not found
        within_cluster_id : int
            idx in the list of each clusters, None if not found
        '''
        label = label.replace('spa-brz', 'por-brz')
        rval = [None, None, None]
        # lang_id
        if label not in speech.nist15_lang_list:
            raise ValueError('Cannot found label:%s' % label)
        rval[0] = np.argmax(label == speech.nist15_lang_list)

        # cluster_id
        for c, x in enumerate(speech.nist15_cluster_lang.iteritems()):
            j = x[1]
            if label in j:
                rval[1] = c
                rval[2] = j.index(label)
        return rval

    # ==================== Timit ==================== #
    timit_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
        'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en',
        'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
        'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
        'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th',
        'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    timit_39 = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd',
        'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
        'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't',
        'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
    timit_map = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
        'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
        'en': 'n', 'nx': 'n',
        'eng': 'ng', 'zh': 'sh', 'ux': 'uw',
        'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil',
        'dcl': 'sil', 'gcl': 'sil', 'h#': 'sil', 'pau': 'sil', 'epi': 'sil'}

    @staticmethod
    def timit_phonemes(phn, map39=False, blank=False):
        ''' Included blank '''
        if type(phn) not in (list, tuple, np.ndarray):
            phn = [phn]
        if map39:
            timit = speech.timit_39
            timit_map = speech.timit_map
            l = 39
        else:
            timit = speech.timit_61
            timit_map = {}
            l = 61

        rphn = []
        for p in phn:
            if p not in timit_map and p not in timit:
                if blank: rphn.append(l)
            else:
                rphn.append(timit.index(timit_map[p]) if p in timit_map else timit.index(p))
        return rphn

    # ==================== Speech Signal Processing ==================== #
    @staticmethod
    def read(f, pcm = False):
        '''
        Return
        ------
            waveform (ndarray), sample rate (int)
        '''
        if pcm or (isinstance(f, str) and 'pcm' in f):
            return np.memmap(f, dtype=np.int16, mode='r')
        return soundfile.read(f)

    @staticmethod
    def preprocess(signal, add_noise=False):
        if len(signal.shape) > 1:
            signal = signal.ravel()

        signal = signal[signal != 0]
        signal = signal.astype(np.float32)
        if add_noise:
            signal = signal + 1e-13 * np.random.randn(signal.shape)
        return signal

    @staticmethod
    def logmel(signal, fs, n_filters=40, n_ceps=13,
            win=0.025, shift=0.01,
            delta1=True, delta2=True, energy=True,
            normalize=True, clean=True,
            vad=True, returnVAD=False):
        import sidekit

        if len(signal.shape) > 1:
            signal = signal.ravel()

        #####################################
        # 1. Some const.
        # n_filters = 40 # The number of mel filter bands
        f_min = 0. # The minimal frequency of the filter bank
        f_max = fs / 2
        # overlap = nwin - int(shift * fs)

        #####################################
        # 2. preprocess.
        if clean:
            signal = speech.preprocess(signal)

        #####################################
        # 3. logmel.
        logmel = sidekit.frontend.features.mfcc(signal,
                        lowfreq=f_min, maxfreq=f_max,
                        nlinfilt=0, nlogfilt=n_filters,
                        fs=fs, nceps=n_ceps, midfreq=1000,
                        nwin=win, shift=shift,
                        get_spec=False, get_mspec=True)
        logenergy = logmel[1]
        logmel = logmel[3].astype(np.float32)

        #####################################
        # 4. delta.
        tmp = [logmel]
        if delta1 or delta2:
            d1 = sidekit.frontend.features.compute_delta(logmel,
                            win=3, method='filter')
            d2 = sidekit.frontend.features.compute_delta(d1,
                            win=3, method='filter')
            if delta1: tmp.append(d1)
            if delta2: tmp.append(d2)
        logmel = np.concatenate(tmp, 1)

        if energy:
            logmel = np.concatenate((logmel, logenergy.reshape(-1, 1)), axis=1)

        #####################################
        # 5. VAD and normalize.
        if vad:
            nwin = int(fs * win)
            idx = sidekit.frontend.vad.vad_snr(signal, 30, fs=fs, shift=shift, nwin=nwin)
            if not returnVAD:
                logmel = logmel[idx, :]

        # Normalize
        if normalize:
            mean = np.mean(logmel, axis = 0)
            var = np.var(logmel, axis = 0)
            logmel = (logmel - mean) / np.sqrt(var)

        if returnVAD and vad:
            return logmel, idx
        return logmel

    @staticmethod
    def mfcc(signal, fs, n_ceps, n_filters=40,
            win=0.025, shift=0.01,
            delta1=True, delta2=True, energy=True,
            normalize=True, clean=True,
            vad=True, returnVAD=False):
        import sidekit

        #####################################
        # 1. Const.
        f_min = 0. # The minimal frequency of the filter bank
        f_max = fs / 2

        #####################################
        # 2. Speech.
        if clean:
            signal = speech.preprocess(signal)

        #####################################
        # 3. mfcc.
        # MFCC
        mfcc = sidekit.frontend.features.mfcc(signal,
                        lowfreq=f_min, maxfreq=f_max,
                        nlinfilt=0, nlogfilt=n_filters,
                        fs=fs, nceps=n_ceps, midfreq=1000,
                        nwin=win, shift=shift,
                        get_spec=False, get_mspec=False)
        logenergy = mfcc[1]
        mfcc = mfcc[0].astype(np.float32)

        if energy:
            mfcc = np.concatenate((mfcc, logenergy.reshape(-1, 1)), axis=1)
        #####################################
        # 4. Add more information.
        tmp = [mfcc]
        if delta1 or delta2:
            d1 = sidekit.frontend.features.compute_delta(mfcc,
                            win=3, method='filter')
            d2 = sidekit.frontend.features.compute_delta(d1,
                            win=3, method='filter')
            if delta1: tmp.append(d1)
            if delta2: tmp.append(d2)
        mfcc = np.concatenate(tmp, 1)
        #####################################
        # 5. Vad and normalize.
        # VAD
        if vad:
            nwin = int(fs * win)
            idx = sidekit.frontend.vad.vad_snr(signal, 30, fs=fs, shift=shift, nwin=nwin)
            if not returnVAD:
                mfcc = mfcc[idx, :]

        # Normalize
        if normalize:
            mean = np.mean(mfcc, axis = 0)
            var = np.var(mfcc, axis = 0)
            mfcc = (mfcc - mean) / np.sqrt(var)

        if returnVAD and vad:
            return mfcc, idx
        return mfcc

    @staticmethod
    def spectrogram(signal, fs, n_ceps=13, n_filters=40,
            win=0.025, shift=0.01,
            normalize=False, clean=True,
            vad=True, returnVAD=False):
        import sidekit

        #####################################
        # 1. Const.
        f_min = 0. # The minimal frequency of the filter bank
        f_max = fs / 2

        #####################################
        # 2. Speech.
        if clean:
            signal = speech.preprocess(signal)

        #####################################
        # 3. mfcc.
        # MFCC
        spt = sidekit.frontend.features.mfcc(signal,
                        lowfreq=f_min, maxfreq=f_max,
                        nlinfilt=0, nlogfilt=n_filters,
                        fs=fs, nceps=n_ceps, midfreq=1000,
                        nwin=win, shift=shift,
                        get_spec=True, get_mspec=False)
        spt = spt[2]
        spt = spt.astype(np.float32)

        #####################################
        # 5. Vad and normalize.
        # VAD
        if vad:
            nwin = int(fs * win)
            idx = sidekit.frontend.vad.vad_snr(signal, 30, fs=fs, shift=shift, nwin=nwin)
            if not returnVAD:
                spt = spt[idx, :]

        # Normalize
        if normalize:
            mean = np.mean(spt, axis = 0)
            var = np.var(spt, axis = 0)
            spt = (spt - mean) / np.sqrt(var)

        if returnVAD and vad:
            return spt, idx
        return spt

    @staticmethod
    def LevenshteinDistance(s1, s2):
        ''' Implementation of the wikipedia algorithm, optimized for memory
        Reference: http://rosettacode.org/wiki/Levenshtein_distance#Python
        '''
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for index2, char2 in enumerate(s2):
            newDistances = [index2 + 1]
            for index1, char1 in enumerate(s1):
                if char1 == char2:
                    newDistances.append(distances[index1])
                else:
                    newDistances.append(1 + min((distances[index1],
                                                 distances[index1 + 1],
                                                 newDistances[-1])))
            distances = newDistances
        return distances[-1]

    @staticmethod
    def LER(y_true, y_pred, return_mean=True):
        ''' This function calculates the Labelling Error Rate (PER) of the decoded
        networks output sequence (out) and a target sequence (tar) with Levenshtein
        distance and dynamic programming. This is the same algorithm as commonly used
        for calculating the word error rate (WER), or phonemes error rate (PER).

        Parameters
        ----------
        y_true : ndarray (nb_samples, seq_labels)
            true values of sequences
        y_pred : ndarray (nb_samples, seq_labels)
            prediction values of sequences

        Returns
        -------
        return : float
            Labelling error rate
        '''
        if not hasattr(y_true[0], '__len__') or isinstance(y_true[0], str):
            y_true = [y_true]
        if not hasattr(y_pred[0], '__len__') or isinstance(y_pred[0], str):
            y_pred = [y_pred]

        results = []
        for ytrue, ypred in zip(y_true, y_pred):
            results.append(speech.LevenshteinDistance(ytrue, ypred) / len(ytrue))
        if return_mean:
            return np.mean(results)
        return results

    @staticmethod
    def decodeMaxOut(y_pred, mask=None):
        if mask is None:
            mask = np.ones(y_pred.shape[:-1])
        mask = mask.astype(np.int8)
        blank = y_pred.shape[-1] - 1
        r = []
        y_pred = T.masked_output(y_pred, mask)
        for y in y_pred:
            y = np.argmax(y, -1).tolist()
            # remove duplicate
            tmp = [y[0]]
            for i in y[1:]:
                if i != tmp[-1]: tmp.append(i)
            # remove blanks, some duplicate may happen again but it is real
            tmp = [i for i in tmp if i != blank]

            r.append(tmp)
        return r

# ======================================================================
# Network
# ======================================================================
class SSH(object):

    """Create a SSH connection object
        Example:
            ssh = SSH('192.168.1.16',username='user',password='pass')
            ssh.ls('.') # same as ls in linux
            ssh.open('/path/to/file') # open stream to any file in remote server
            ssh.get_file('/path/to/file') # read the whole file in remote server
            ssh.close()
    """

    def __init__(self, hostname, username, password=None, pkey_path=None, port=22):
        super(SSH, self).__init__()

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        k = None
        if pkey_path:
            k = paramiko.RSAKey.from_private_key_file(pkey_path)
        ssh.connect(hostname=hostname,
                    username=username, port=port,
                    password=password,
                    pkey = k)
        sftp = ssh.open_sftp()
        self.ssh = ssh
        self.sftp = sftp

    def _file_filter(self, fname):
        if len(fname) == 0 or fname == '.' or fname == '..':
            return False
        return True

    def ls(self, path='.'):
        sin, sout, serr = self.ssh.exec_command('ls -a ' + path)
        file_list = sout.read()
        file_list = [f for f in file_list.split('\n') if self._file_filter(f)]
        return file_list

    def open(self, fpaths, mode='r', bufsize=-1):
        if not (isinstance(fpaths, list) or isinstance(fpaths, tuple)):
            fpaths = [fpaths]
        results = []
        for f in fpaths:
            try:
                results.append(self.sftp.open(f, mode=mode, bufsize=bufsize))
            except:
                pass
        if len(results) == 1:
            return results[0]
        return results

    def get_file(self, fpaths, bufsize=-1):
        if not (isinstance(fpaths, list) or isinstance(fpaths, tuple)):
            fpaths = [fpaths]
        results = []
        for f in fpaths:
            try:
                results.append(self.sftp.open(f, mode='r', bufsize=bufsize))
            except:
                pass
        if len(results) == 1:
            return results[0]
        return results

    def isdir(self, path):
        try:
            return S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            #Path does not exist, so by definition not a directory
            return None

    def getwd(self):
        ''' This method may return NONE '''
        return self.sftp.getcwd()

    def setwd(self, path):
        self.sftp.chdir(path)

    def mkdir(self, path, mode=511):
        self.sftp.mkdir(path, mode)

    def close(self):
        self.sftp.close()
        self.ssh.close()

# ======================================================================
# Net
# ======================================================================
from six.moves.urllib.request import FancyURLopener

class ParanoidURLopener(FancyURLopener):

    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise Exception('URL fetch failure on {}: {} -- {}'.format(url, errcode, errmsg))

class net():

    @staticmethod
    def get_file(fname, origin):
        ''' Get file from internet or local network.

        Parameters
        ----------
        fname : str
            name of downloaded file
        origin : str
            html link, path to file want to download

        Returns
        -------
        return : str
            path to downloaded file

        Notes
        -----
        Download files are saved at one of these location (order of priority):
         - ~/.dnntoolkit/datasets/
         - /tmp/.dnntoolkit/datasets/
        '''
        if os.path.exists(origin) and not os.path.isdir(origin):
            return origin
        import pwd
        user_name = pwd.getpwuid(os.getuid())[0]

        datadir_base = os.path.expanduser(os.path.join('~', '.dnntoolkit'))
        if not os.access(datadir_base, os.W_OK):
            datadir_base = os.path.join('/tmp', user_name, '.dnntoolkit')
        datadir = os.path.join(datadir_base, 'datasets')
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        fpath = os.path.join(datadir, fname)

        if not os.path.exists(fpath):
            logger.info('Downloading data from', origin)
            global progbar
            progbar = None

            def dl_progress(count, block_size, total_size):
                logger.progress(count * block_size, total_size,
                    title='Downloading %s' % fname, newline=False,
                    idx='downloading')

            ParanoidURLopener().retrieve(origin, fpath, dl_progress)
            progbar = None

        return fpath
