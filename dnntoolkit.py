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
import time
import warnings
from stat import S_ISDIR

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
    def preprocess_mpi(jobs_list, features_func, save_func, n_cache=30):
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
            print('Process 0 found %d jobs' % len(jobs_list))
            jobs = mpi.segment_job(jobs_list, npro)
            n_loop = max([len(i) for i in jobs])
        else:
            jobs = None
            n_loop = 0
            print('Process %d waiting for Process 0!' % rank)
        comm.Barrier()

        jobs = comm.scatter(jobs, root=0)
        n_loop = comm.bcast(n_loop, root=0)
        print('Process %d receive %d jobs' % (rank, len(jobs)))

        #####################################
        # 2. Start preprocessing.
        data = []

        for i in xrange(n_loop):
            if i % n_cache == 0 and i > 0:
                all_data = comm.gather(data, root=0)
                if rank == 0:
                    print('Saving data at process 0')
                    all_data = [k for j in all_data for k in j]
                    if len(all_data) > 0:
                        save_func(all_data)
                data = []

            if i >= len(jobs): continue
            feature = features_func(jobs[i])
            if feature is not None:
                data.append(feature)

            if i % 50 == 0:
                print('Rank:%d preprocessed %d files!' % (rank, i))

        all_data = comm.gather(data, root=0)
        if rank == 0:
            print('Saving data before exit !!!!\n')
            all_data = [k for j in all_data for k in j]
            if len(all_data) > 0:
                save_func(all_data)


# ======================================================================
# io helper
# ======================================================================
class io():

    @staticmethod
    def extract_files(path, filter_func=None):
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
    def find_in_module(module, identifier):
        import six
        if isinstance(module, six.string_types):
            module = globals()[module]

        from inspect import getmembers
        for i in getmembers(module):
            if identifier in i:
                return i[1]
        return None

# ======================================================================
# Array Utils
# ======================================================================
class tensor():
    # ======================================================================
    # Sequence processing
    # ======================================================================

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
            > print(split_chunks(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 5, 1))
            > [[1, 2, 3, 4, 5],
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

    # ======================================================================
    # Theano
    # ======================================================================

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
            Input: [[0.0, 0.0, 0.5],
                    [0.0, 0.3, 0.1],
                    [0.6, 0.0, 0.2]]

            Output: [[0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0]]
        '''
        return T.cast(T.eq(T.arange(x.shape[1])[None, :], T.argmax(x, axis=1, keepdims=True)), theano.config.floatX)

    @staticmethod
    def on_gpu():
        return theano.config.device[:3] == 'gpu'

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

# ======================================================================
# Early stop
# ======================================================================
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
    '''
    gl_exit_threshold = threshold

    if len(validation) == 0:
        return 0, 0
    shouldStop = 0
    shouldSave = 0

    gl_t = 100 * (validation[-1] / min(validation) - 1)
    if gl_t == 0: # min = current_value
        shouldSave = 1
        shouldStop = -1
    elif gl_t > gl_exit_threshold:
        shouldStop = 1
        shouldSave = -1

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

# ====== early stop ====== #
def earlystop(costs, generalization_loss = False, generalization_sensitive=False, hope_hop=False, threshold=None):
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

# ======================================================================
# Model
# ======================================================================
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
        h._name = 'Merge:<' + ','.join(all_names) + '>'
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
        print(fmt % sep)
        print(fmt % ('Time', 'Tags', 'Values'))
        print(fmt % sep)
        # contents
        for row in self._history:
            row = tuple([str(i) for i in row])
            print(fmt % row)

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

class model(object):

    """docstring for Model
    """

    def __init__(self):
        super(model, self).__init__()
        self._history = []
        self._working_history = None

        self._weights = []
        self._save_path = None

        self._model_func = None
        self._model_name = None
        self._model_args = None
        self._api = 'lasagne'
        self._sandbox = ''

        # contain real model object
        self._model = None

    # ==================== Model manager ==================== #
    def set_weights(self, weights):
        self._weights = []
        for w in weights:
            self._weights.append(w.astype(np.float32))

    def get_weights(self):
        return self._weights

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

        primitive = (bool, int, float, str)
        sandbox = {}

        if 'lasagne' in api:
            self._api = 'lasagne'
            self._model_func = model
            for k, v in model.func_globals.items():
                if isinstance(v, primitive):
                    sandbox[k] = v
        elif 'keras' in api:
            self._api = 'keras'
            raise NotImplementedError()
        elif 'blocks' in api:
            self._api = 'blocks'
            raise NotImplementedError()
        #H5py not support binary string, cannot use marshal
        self._sandbox = sandbox
        self._model_args = kwargs
        self._model_name = model.func_name

    def create_model(self):
        if self._model_func is None:
            raise ValueError("You must save_model first")
        if self._model is None:
            func = self._model_func
            args = self._model_args
            print('*** INFO: creating network ... ***')
            self._model = func(**args)

            if self._api == 'lasagne':
                import lasagne
                # load old weight
                if len(self._weights) > 0:
                    try:
                        lasagne.layers.set_all_param_values(self._model, self._weights)
                        print('*** INFO: successfully load old weights ***')
                    except Exception, e:
                        print('*** WARNING: Cannot load old weights ***')
                        print(str(e))
                        import traceback; traceback.print_exc();
            else:
                raise NotImplementedError()
        return self._model

    def pred(self, *X):
        self.create_model()

        # ====== Create prediction function ====== #
        if self._pred is None:
            if self._api == 'lasagne':
                # create prediction function
                input_layers = lasagne.layers.find_layers(self._model, types=lasagne.layers.InputLayer)
                input_var = [l.input_var for l in input_layers]
                self._pred = theano.function(
                    inputs=input_var,
                    outputs=lasagne.layers.get_output(self._model, deterministic=True),
                    allow_input_downcast=True,
                    on_unused_input=None)
            else:
                raise NotImplementedError

        # ====== make prediction ====== #
        prediction = None
        try:
            prediction = self._pred(*X)
        except Exception, e:
            print('*** ERROR: Cannot make prediction ***')
            if self._api == 'lasagne':
                import lasagne
                input_layers = lasagne.layers.find_layers(self._model, types=lasagne.layers.InputLayer)
                print('Input order:' + str([l.name for l in input_layers]))
            print(str(e))
            import traceback; traceback.print_exc();
        return prediction

    def print_model(self):
        import inspect
        if self._model_func is not None:
            print(inspect.getsource(self._model_func))

    # ==================== History manager ==================== #
    def _check_current_working_history(self):
        if self._working_history is None:
            if len(self._history) == 0:
                self._history.append(_history())
            self._working_history = self._history[-1]

    def __getitem__(self, key):
        self._check_current_working_history()
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
        raise ValueError('Model index must be [slice], [int] or [str]')

    def new_frame(self, name=None, description=None):
        self._history.append(_history(name, description))
        self._working_history = self._history[-1]

    def drop_frame(self):
        if len(self._history) < 2:
            self._history = []
        else:
            self._history = self._history[:-1]
        self._working_history = None
        self._check_current_working_history()

    def record(self, values, *tags):
        self._check_current_working_history()
        self._working_history.record(values, *tags)

    def update(self, tags, new, after=None, before=None, n=None, absolute=False):
        self._check_current_working_history()
        self._working_history.update(self, tags, new,
            after, before, n, absolute)

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
        self._check_current_working_history()
        return self._working_history.select(tags, default, after, before, n,
            filter_value, absolute, newest, return_time)

    def print_history(self):
        self._check_current_working_history()
        self._working_history.print_history()

    def print_frames(self):
        self._check_current_working_history()
        for i in self._history:
            if i == self._working_history:
                print('* ' + str(i))
            else:
                print(i)

    def __str__(self):
        import inspect

        s = ''
        s += 'Model: %s' % self._save_path + '\n'

        # history
        self._check_current_working_history()
        for i in self._history:
            if i == self._working_history:
                s += '* ' + str(i) + '\n'
            else:
                s += i + '\n'

        # weight
        s += '======== Weights ========\n'
        for w in self._weights:
            s += ' - shape:%s' % w.shape + '\n'

        # model function
        s += '======== Code ========\n'
        s += ' - api:%s' % self._api + '\n'
        s += ' - name:%s' % self._model_name + '\n'
        s += ' - args:%s' % str(self._model_args) + '\n'
        s += ' - sandbox:%s' % str(self._sandbox) + '\n'
        if self._model_func is not None:
            s += ' - code:\n%s' % inspect.getsource(self._model_func.func_code) + '\n'
        else:
            s += ' - code:\n%s' % self._model_func + '\n'

        return s

    # ==================== Load & Save ==================== #
    def save(self, path=None):
        if path is None and self._save_path is None:
            raise ValueError("Save path haven't specified!")
        path = path if path is not None else self._save_path
        self._save_path = path

        import cPickle
        import marshal
        from array import array

        f = h5py.File(path, 'w')
        f['history'] = cPickle.dumps(self._history)

        if self._model_func is not None:
            model_func = marshal.dumps(self._model_func.func_code)
            b = array("B", model_func)
            f['model_func'] = cPickle.dumps(b)
            f['model_args'] = cPickle.dumps(self._model_args)
            f['model_name'] = self._model_name
            f['sandbox'] = cPickle.dumps(self._sandbox)
            f['api'] = self._api

        if len(self._weights) > 0:
            for i, w in enumerate(self._weights):
                f['weight_%d' % i] = w
            f['nb_weights'] = len(self._weights)
        f.close()

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            m = model()
            m._save_path = path
            return m
        import cPickle
        import marshal
        import types

        m = model()
        m._save_path = path

        f = h5py.File(path, 'r')
        if 'history' in f:
            m._history = cPickle.loads(f['history'].value)
        else:
            m._history = []

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
        else: m._sandbox = None

        if m._api is not None and m._sandbox is not None:
            globals()[m._api] = __import__(m._api)
            for k, v in m._sandbox.iteritems():
                globals()[k] = v

        if 'model_func' in f:
            b = cPickle.loads(f['model_func'].value)
            m._model_func = marshal.loads(b.tostring())
            m._model_func = types.FunctionType(m._model_func, globals(), m._model_name)
        else: m._model_func = None

        if 'nb_weights' in f:
            for i in xrange(f['nb_weights'].value):
                m._weights.append(f['weight_%d' % i].value)

        f.close()
        return m

# ======================================================================
# Trainer
# ======================================================================
def _callback(trainer):
    pass

def seed_generator(seed, size=30000):
    np.random.seed(seed)
    random_seed = np.random.randint(0, 10e8, size=size)
    for i in xrange(size):
        yield random_seed[i]

def _parse_data_config(task, data):
    '''return train,valid,test'''
    train = None
    test = None
    valid = None
    if type(data) in (tuple, list):
        if type(data[0]) not in (tuple, list):
            if 'train' in task: train = data
            elif 'test' in task: test = data
            elif 'valid' in task: valid = data
        else:
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

    return train, valid, test

class trainer(object):

    """
    TODO: request validation function, add custome data (not instance of dataset)
    Value can be queried on callback:
        idx: current run idx in the strategies
        cost: current training, testing, validating cost
        iter: number of iteration
        data: current data (batch_start)
        epoch: current epoch
        task: current task
    """

    def __init__(self):
        super(trainer, self).__init__()
        self._seed = seed_generator(MAGIC_SEED)
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

        self._epoch_start = _callback
        self._epoch_end = _callback
        self._batch_start = _callback
        self._batch_end = _callback
        self._train_end = _callback
        self._valid_end = _callback
        self._test_end = _callback

        self._stop = False

        self._log_enable = True
        self._log_newline = False

    # ==================== Command ==================== #
    def stop(self):
        ''' Stop current activity of this trainer immediatelly '''
        self._stop = True

    # ==================== Setter ==================== #
    def set_log(self, enable=True, newline=False):
        self._log_enable = enable
        self._log_newline = newline

    def set_dataset(self, data, train=None, valid=None, test=None):
        ''' Set dataset for trainer.

        Parameters
        ----------
        data : dnntoolkit.dataset
            dataset instance which contain all your data
        train : str, list(str)
            list of dataset used for training
        valid : str, list(str)
            list of dataset used for validation
        test : str, list(str)
            list of dataset used for testing

        Returns
        -------
        return : trainer
            for chaining method calling

        Note
        ----
        the order of train, valid, test must be the same in model function
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
                     train_end=_callback,
                     valid_end=_callback,
                     test_end=_callback):
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

        self._train_end = train_end
        self._valid_end = valid_end
        self._test_end = test_end
        return self

    def set_strategy(self, task=None, data=None, epoch=1, batch=512, validfreq=20, shuffle=True, seed=None, yaml=None):
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
        batch : int
            number of samples for each batch
        validfreq : int, float(0.-1.)
            validation frequency when training, when float, it mean percentage
            of dataset
        shuffle : boolean
            shuffle dataset while training
        seed : int
            set seed for shuffle so the result is reproducible
        yaml : str
            path to yaml strategy file. When specify this arguments,
            all other arguments are ignored

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
                if 'seed' in s: self._seed = seed_generator(seed)
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
            'validfreq': validfreq
        })
        if seed is not None:
            self._seed = seed_generator(seed)
        return self

    # ==================== Logic ==================== #
    def _early_stop(self):
        # just a function reset stop function and return its value
        tmp = self._stop
        self._stop = False
        return tmp

    def _create_iter(self, names, batch, shuffle):
        seed = self._seed.next()
        data = [self._dataset[i].iter(batch, shuffle=shuffle, seed=seed) for i in names]
        return enumerate(zip(*data))

    def _cost(self, task, valid_data, batch):
        self.task = task

        n_samples = self._dataset[valid_data[0]].shape[0]
        valid_cost = []
        n = 0
        it = 0
        for i, data in self._create_iter(valid_data, batch, False):
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
                    title='%s:Cost:%.2f' % (task, np.mean(cost)),
                    newline=self._log_newline)

            # batch end
            self.cost = cost
            self.iter = it
            self._batch_end(self)
            self.data = None
            self.cost = None

        # ====== callback ====== #
        self.cost = valid_cost
        if task == 'valid':
            self._valid_end(self) # callback
        else:
            self._test_end(self)

        # ====== statistic of validation ====== #
        valid_mean = np.mean(self.cost)
        valid_median = np.median(self.cost)
        valid_min = np.percentile(self.cost, 5)
        valid_max = np.percentile(self.cost, 95)
        valid_var = np.var(self.cost)
        print('Statistic: Mean:%.2f Var:%.2f Med:%.2f Min:%.2f Max:%.2f' %
            (valid_mean, valid_var, valid_median, valid_min, valid_max))

        # ====== reset all flag ====== #
        self.cost = None
        self.task = None
        self.iter = 0

    def _finish_train(self, train_cost):
        self.cost = train_cost
        self._train_end(self) # callback
        self.cost = None
        self.task = None
        self.it = 0

    def _train(self, train_data, valid_data, epoch, batch, validfreq, shuffle):
        self.task = 'train'

        self.iter = 0
        it = 0
        ntrain = self._dataset[train_data[0]].shape[0]
        if validfreq < 1.0: # validate validfreq
            validfreq = int(max(validfreq * ntrain / batch, 1))
        train_cost = []
        # ====== start ====== #
        for i in xrange(epoch):
            self.epoch = i
            self.iter = it
            self._epoch_start(self) # callback
            if self._early_stop(): # earlystop
                self._finish_train(train_cost)
                return
            epoch_cost = []
            n = 0
            # ====== start batches ====== #
            for j, data in self._create_iter(train_data, batch, shuffle):
                # start batch
                n += data[0].shape[0]
                it += 1
                self.data = data
                self.iter = it
                self._batch_start(self) # callback

                # main updates
                cost = self._updates_func(*self.data)

                # log
                epoch_cost.append(cost)
                train_cost.append(cost)
                if self._log_enable:
                    logger.progress(n, max_val=ntrain,
                        title='Epoch:%d,Iter:%d,Cost:%.2f' % (i + 1, it, cost),
                        newline=self._log_newline)

                # end batch
                self.cost = cost
                self.iter = it
                self._batch_end(self)  # callback
                self.data = None
                self.cost = None
                if self._early_stop(): # earlystop
                    self._finish_train(train_cost)
                    return

                # validation
                if it > 0 and it % validfreq == 0:
                    if valid_data is not None:
                        self._cost('valid', valid_data, batch)
                        if self._early_stop(): # earlystop
                            self._finish_train(train_cost)
                            return
                    self.task = 'train' # restart flag back to train

            # end epoch
            self.cost = epoch_cost
            self.iter = it
            self._epoch_end(self) # callback
            self.cost = None
            if self._early_stop(): # earlystop
                self._finish_train(train_cost)
                return

        # end training
        self._finish_train(train_cost)

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

                epoch = config['epoch']
                batch = config['batch']
                validfreq = config['validfreq']
                shuffle = config['shuffle']

                self.idx += 1
                if self._log_enable:
                    print('\n******* %d-th run, with configuration: *******' % self.idx)
                    print(' - Task:%s' % task)
                    print(' - Train data:%s' % str(train))
                    print(' - Valid data:%s' % str(valid))
                    print(' - Test data:%s' % str(test))
                    print(' - Epoch:%d' % epoch)
                    print(' - Batch:%d' % batch)
                    print(' - Validfreq:%d' % validfreq)
                    print(' - Shuffle:%s' % str(shuffle))
                    print('**********************************************')

                if 'train' in task:
                    if train is None:
                        print('*** WARNING: no TRAIN data found, ignored **')
                    else:
                        self._train(train, valid, epoch, batch, validfreq, shuffle)
                elif 'valid' in task:
                    if valid is None:
                        print('*** WARNING: no VALID data found, ignored **')
                    else:
                        self._cost('valid', valid, batch)
                elif 'test' in task:
                    if test is None:
                        print('*** WARNING: no TEST data found, ignored **')
                    else:
                        self._cost('test', test, batch)
        except Exception, e:
            print(str(e))
            import traceback; traceback.print_exc();
            return False
        return True

    # ==================== Debug ==================== #
    def __str__(self):
        s = '\n'
        s += 'Dataset:' + str(self._dataset) + '\n'
        s += 'Current run:%d' % self.idx + '\n'
        s += '============ \n'
        s += 'defTrain:' + str(self._train_data) + '\n'
        s += 'defValid:' + str(self._valid_data) + '\n'
        s += 'defTest:' + str(self._test_data) + '\n'
        s += '============ \n'
        s += 'Cost_func:' + str(self._cost_func) + '\n'
        s += 'Updates_func:' + str(self._updates_func) + '\n'
        s += '============ \n'
        s += 'Epoch start:' + str(self._epoch_start) + '\n'
        s += 'Epoch end:' + str(self._epoch_end) + '\n'
        s += 'Batch start:' + str(self._batch_start) + '\n'
        s += 'Batch end:' + str(self._batch_end) + '\n'
        s += 'Train end:' + str(self._train_end) + '\n'
        s += 'Valid end:' + str(self._valid_end) + '\n'
        s += 'Test end:' + str(self._test_end) + '\n'

        for i, st in enumerate(self._strategy):
            train, valid, test = _parse_data_config(st['task'], st['data'])
            if train is None: train = self._train_data
            if test is None: test = self._test_data
            if valid is None: valid = self._valid_data

            s += '====== Strategy %d-th ======\n' % (i + 1)
            s += ' - Task:%s' % st['task'] + '\n'
            s += ' - Train:%s' % str(train) + '\n'
            s += ' - Valid:%s' % str(valid) + '\n'
            s += ' - Test:%s' % str(test) + '\n'
            s += ' - Epoch:%d' % st['epoch'] + '\n'
            s += ' - Batch:%d' % st['batch'] + '\n'
            s += ' - Shuffle:%s' % st['shuffle'] + '\n'

        return s


# ======================================================================
# Data Preprocessing
# ======================================================================
def create_batch(n_samples, batch_size, start=None, end=None, prng=None, upsample=None):
    '''
    Parameters
    ----------
    prng : numpy.random.RandomState
        if prng != None, the upsampling process will be randomized
    upsample : int
        upsample > n_samples, batch will be sampled from original data to make
        the same total number of sample

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
    if end is None or end < start or end > n_samples:
        end = n_samples
    if start < 1.0:
        start = int(start * n_samples)
    if end <= 1.0:
        end = int(end * n_samples)
    n_samples = end - start

    if upsample is None or upsample < n_samples:
        upsample = n_samples

    #####################################
    # 2. Init.
    idx = range(start, end)
    n_batch = max(int(math.floor(n_samples / batch_size)), 1)
    jobs = mpi.segment_job(idx, n_batch)
    jobs = [(j[0], j[-1] + 1) for j in jobs if len(j) > 0]

    #####################################
    # 3. Upsample jobs.
    upsample_jobs = []
    n = n_samples
    i = 0
    while n < upsample:
        # pick a package
        if prng is None:
            added_job = jobs[i % len(jobs)]
            i += 1
        elif prng is not None:
            added_job = jobs[prng.randint(0, len(jobs))]
        # remove redundant size
        tmp = added_job[1] - added_job[0]
        if n + tmp > upsample:
            tmp = n + tmp - upsample
            added_job = (added_job[0], added_job[1] - tmp)
        n += added_job[1] - added_job[0]
        # done
        upsample_jobs.append(added_job)
    return jobs + upsample_jobs

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

class _batch(object):

    """docstring for _batch"""

    def __init__(self, key, hdf):
        super(_batch, self).__init__()
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
        return np.concatenate([i.value for i in self._data], axis=0)

    # ==================== Arithmetic ==================== #
    def sum2(self, axis=0):
        ''' sum(X^2) '''
        self._is_dataset_init()

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
        for d in self._data:
            _hdf5_append_to_dataset(d, other)
        return self

    def duplicate(self, other):
        if not isinstance(other, int):
            raise TypeError('Only duplicate by int factor')
        if len(self._data) == 0:
            raise TypeError("Data haven't initlized yet")
        for d in self._data:
            copy = d[:]
            for i in xrange(other - 1):
                _hdf5_append_to_dataset(d, copy)
        return self

    def _iter_fast(self, ds, batch_size, start=None, end=None,
            shuffle=True, seed=None, normalizer=None):
        # craete random seed
        prng1 = None
        prng2 = _dummy_shuffle
        if shuffle:
            if seed is None:
                seed = MAGIC_SEED
            prng1 = RandomState(seed)
            prng2 = RandomState(seed)

        batches = create_batch(ds.shape[0], batch_size,
            start, end, prng1)
        prng2.shuffle(batches)

        for i, j in batches:
            data = ds[i:j]
            prng2.shuffle(data) # this will slow thing a little bit
            if normalizer is None:
                yield data
            else:
                yield normalizer(data)

    def _iter_slow(self, batch_size=128, start=None, end=None,
        shuffle=True, seed=None, normalizer=None, mode=0):
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
        if mode == 1:
            maxsize = max(all_size)
            all_batch_size = \
                [int(math.ceil(batch_size / n_dataset)) for i in xrange(n_dataset)]
            all_upsample = [maxsize for i in xrange(n_dataset)]
        elif mode == 2:
            s = sum(all_size)
            all_batch_size = \
                [int(math.ceil(batch_size * float(i / s))) for i in all_size]
            all_upsample = all_size
        else:
            all_batch_size = [batch_size for i in xrange(n_dataset)]
            all_upsample = all_size

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
        if mode == 1 or mode == 2:
            for i in zip_longest(*tmp_block_batch):
                all_block_batch.append([(k, v) for k, v in enumerate(i) if v is not None])
        else:
            for i, batches in enumerate(tmp_block_batch):
                for b in batches:
                    all_block_batch.append( # i=dataset_idx; b=(batches)
                        [(i, b)]
                    )
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
            prng2.shuffle(batches)
            if normalizer is None:
                yield batches
            else:
                yield normalizer(batches)

    def iter(self, batch_size=128, start=None, end=None,
        shuffle=True, seed=None, normalizer=None, mode=0):
        ''' Create iteration for all dataset contained in this _batch

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
            0 - default, read one by one each dataset
            1 - equally read each dataset, upsampling smaller dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size << dataset size)
            2 - proportionately read each dataset (e.g. batch_size=512,
                dataset1_size=1000, dataset2_size=500 => ds1=341, ds2=170)

        Returns
        -------
        return : generator
            generator generate batches of data

        Notes
        -----
        This method is thread-safe, as it uses private instance of RandomState,
        the order will be preserved if using iter in multi thread.
        To create consistent permutation of shuffled dataset, you must:
         - both batch have the same number and order of dataset
         - using the same seed and mode when calling iter()
         Hint: small level of batch shuffle can be obtained by using normalizer
         function
        '''
        self._is_dataset_init()
        if batch_size == 'auto':
            batch_size = 128

        if len(self._data) == 1:
            return self._iter_fast(self._data[0], batch_size, start, end,
                                   shuffle, seed, normalizer)
        else:
            return self._iter_slow(batch_size, start, end, shuffle, seed,
                                   normalizer, mode)

    def __len__(self):
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
        for k, d in zip(self._key, self._data):
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
                if f[i].dtype == obj_type:
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
            warnings.warn('Have not set write mode, default is [last]', RuntimeWarning)
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
        if type(key) not in (tuple, list):
            key = [key]

        # ====== Return object ====== #
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
        batch = _batch(keys, ret)
        self._datamap[idx] = batch
        return batch

    def __setitem__(self, key, value):
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
            print(hdf)
            for k in key:
                for h in hdf:
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
            for hdf in self._hdf:
                s += '======== %s ========' % hdf.filename + '\n'
                all_data = _hdf5_get_all_dataset(hdf)
                all_data = [(d, str(hdf[d].shape), str(hdf[d].dtype)) for d in all_data]
                for i in all_data:
                    s += ' - name:%-13s  shape:%-18s  dtype:%s' % i + '\n'
        return s[:-1]

    # ==================== Static loading ==================== #
    @staticmethod
    def load_mnist(path='https://s3.amazonaws.com/ai-datasets/mnist.hdf'):
        '''
        path : str
            local path or url to hdf5 datafile
        '''
        datapath = net.get_file('mnist.hdf', path)
        print('Loading data from: %s' % datapath)
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
    def confusion_matrix(cm, labels, axis=None):
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
        axis.set_xticklabels(labels, rotation=90)
        axis.set_yticklabels(labels)

        axis.set_ylabel('True label')
        axis.set_xlabel('Predicted label')
        # axis.tight_layout()
        return axis

    @staticmethod
    def spectrogram(s, ax=None):
        from matplotlib import pyplot as plt

        if s.shape[0] == 1 or s.shape[1] == 1:
            pass

        ax = ax if ax is not None else plt.gca()
        if s.shape[0] > s.shape[1]:
            s = s.T
        img = ax.imshow(s, cmap=plt.cm.Blues, interpolation='bilinear',
            origin="lower", aspect="auto")
        plt.colorbar(img, ax=ax)
        return ax

    @staticmethod
    def bar_str(arr, max_arr=None):
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
        return rval[:-1]

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

class logger():
    _last_value = 0
    _last_time = -1
    """docstring for Logger"""
    @staticmethod
    def progress(p, max_val=1.0, title='Progress', bar='=', newline=False):
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

        if p >= max_val:
            sys.stdout.write("\n")

    @staticmethod
    def create_logger(logging_path, multiprocess=False):
        import logging
        # saving path
        fh = logging.FileHandler(logging_path, mode='w')
        fh.setFormatter(logging.Formatter(
            fmt = '%(asctime)s %(levelname)s  %(message)s',
            datefmt = '%d/%m/%Y %I:%M:%S'))
        fh.setLevel(logging.DEBUG)
        # print
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(
            fmt = '%(asctime)s %(levelname)s  %(message)s',
            datefmt = '%d/%m/%Y %I:%M:%S'))
        sh.setLevel(logging.DEBUG)
        # add the current logger
        logging.getLogger().setLevel(logging.NOTSET)
        logging.getLogger().addHandler(sh)
        if multiprocess:
            import multiprocessing_logging
            logging.getLogger().addHandler(multiprocessing_logging.MultiProcessingHandler('mpi', fh))
        else:
            logging.getLogger().addHandler(fh)
        return logging.getLogger()

# ======================================================================
# Feature
# ======================================================================
class speech():

    """docstring for speech"""

    def __init__(self):
        super(speech, self).__init__()

    # ======================================================================
    # Predefined datasets information
    # ======================================================================
    nist15_cluster_lang = OrderedDict([
        ['ara', ['ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb']],
        ['zho', ['zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu']],
        ['eng', ['eng-gbr', 'eng-usg', 'eng-sas']],
        ['fre', ['fre-waf', 'fre-hat']],
        ['qsl', ['qsl-pol', 'qsl-rus']],
        ['spa', ['spa-car', 'spa-eur', 'spa-lac', 'spa-brz']]
    ])
    nist15_lang_list = np.asarray([
        'ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb',
        'zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu',
        'eng-gbr', 'eng-usg', 'eng-sas',
        'fre-waf', 'fre-hat',
        'qsl-pol', 'qsl-rus',
        'spa-car', 'spa-eur', 'spa-lac', 'spa-brz'])
    nist15_within_cluster = {
        'ara-arz': 0, 'ara-acm': 1, 'ara-apc': 2, 'ara-ary': 3, 'ara-arb': 4,
        'zho-yue': 0, 'zho-cmn': 1, 'zho-cdo': 2, 'zho-wuu': 3,
        'eng-gbr': 0, 'eng-usg': 1, 'eng-sas': 2,
        'fre-waf': 0, 'fre-hat': 1,
        'qsl-pol': 0, 'qsl-rus': 1,
        'spa-car': 0, 'spa-eur': 1, 'spa-lac': 2, 'spa-brz': 3
    }

    @staticmethod
    def timit_phonemes(p, map39=False):
        ''' Mapping from 61 classes to 39 classes '''
        phonemes = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
            'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',
            'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy',
            'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau',
            'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v',
            'w', 'y', 'z', 'zh']
        return phonemes.index(p)

    @staticmethod
    def nist15_label(label, lang=False, cluster=False, within_cluster=False):
        label = label.replace('por-', 'spa-')
        rval = []
        if lang:
            for i, j in enumerate(speech.nist15_lang_list):
                if j in label:
                    rval.append(i)
        if cluster:
            for i, j in enumerate(speech.nist15_cluster_lang.keys()):
                if j in label:
                    rval.append(i)
        if within_cluster:
            for i in speech.nist15_within_cluster.keys():
                if i in label:
                    rval.append(speech.nist15_within_cluster[i])
        if len(rval) == 1:
            rval = rval[0]
        return rval

    # ======================================================================
    # Speech Signal Processing
    # ======================================================================
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
        logmel = logmel[3]

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
            logmel = logmel.astype(np.float32)
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
        mfcc = mfcc[0]

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
            mfcc = mfcc.astype(np.float32)
            mean = np.mean(mfcc, axis = 0)
            var = np.var(mfcc, axis = 0)
            mfcc = (mfcc - mean) / np.sqrt(var)

        if returnVAD and vad:
            return mfcc, idx
        return mfcc

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
            print('Downloading data from', origin)
            global progbar
            progbar = None

            def dl_progress(count, block_size, total_size):
                logger.progress(count * block_size, total_size,
                    title='Downloading %s' % fname)

            ParanoidURLopener().retrieve(origin, fpath, dl_progress)
            progbar = None

        return fpath
