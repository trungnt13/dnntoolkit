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
# ======================================================================
from __future__ import print_function, division

import os
import sys
import math
import time

from itertools import izip
from collections import OrderedDict

import numpy as np
import scipy as sp

import theano
from theano import tensor

import h5py

import pandas as pd

import soundfile

import paramiko
from stat import S_ISDIR

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

# ======================================================================
# io helper
# ======================================================================
class io():

	@staticmethod
	def extract_files(path, filter_func=None):
		''' Recurrsively get all files in the given path '''
		file_list = []
		for p in os.listdir(path):
			p = os.path.join(path, p)
			if os.path.isdir(p):
				f = io.extract_files(p, filter_func)
				if f is not None: file_list += f
			else:
				if filter_func is not None and not filter_func(p):
					continue
				file_list.append(p)
		return file_list

# ======================================================================
# Array Utils
# ======================================================================
class T():

	@staticmethod
	def pad_sequences(sequences, maxlen=None, dtype='int32',
						padding='pre', truncating='pre', value=0.):
	    """
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
		x: numpy array with dimensions (number_of_sequences, maxlen)

	    """
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
		for x, mask in izip(X, X_mask):
			x = x[np.nonzero(mask)]
			res.append(x.tolist())
		return res

	@staticmethod
	def to_categorical(y, nb_classes=None):
	    '''Convert class vector (integers from 0 to nb_classes)
	    to binary class matrix, for use with categorical_crossentropy
	    '''
	    y = np.asarray(y, dtype='int32')
	    if not nb_classes:
	        nb_classes = np.max(y) + 1
	    Y = np.zeros((len(y), nb_classes))
	    for i in range(len(y)):
	        Y[i, y[i]] = 1.
	    return Y

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
		return theano.shared(np.cast[dtype](val))

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
# Model
# ======================================================================
def _create_comparator(t):
	return lambda x: x == t

def _is_tags_match(func, tags, absolute=False):
	'''
	Example
	-------
	>>> tags = [1, 2, 3]
	>>> func = [lambda x: x == 1]
	>>> func1 = [lambda x: x == 1, lambda x: x == 2, lambda x: x == 3]
	>>> _is_tags_match(func, tags, absolute=False) # True
	>>> _is_tags_match(func, tags, absolute=True) # False
	>>> _is_tags_match(func1, tags, absolute=True) # True
	'''
	for f in func:
		match = False
		for t in tags:
			match |= f(t)
		if not match: return False
	if absolute and len(func) != len(tags):
		return False
	return True

def _check_gs(validation):
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

def _check_gl(validation):
	gl_exit_threshold = 3

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

# TODO: Add implementation for loading the whole models
class Model(object):

	"""docstring for Model
	"""

	def __init__(self):
		super(Model, self).__init__()
		self._history = []
		self._weights = []
		self._save_path = None

		self._model_func = ''
		self._api = 'lasagne'
		self._sandbox = ''

		self._model = None

	# ==================== Model manager ==================== #
	def set_weights(self, weights):
		self._weights = []
		for w in weights:
			self._weights.append(w)

	def get_weights(self):
		return self._weights

	def save_model(self, model, api):
		'''
		model: is a callable() without any arguments, return model when called
		'''
		import marshal
		import cPickle
		primitive = (bool, int, float, str)
		sandbox = {}

		if 'lasagne' in api:
			self._api = 'lasagne'
			self._model_func = marshal.dumps(model.func_code)
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
		self._sandbox = cPickle.dumps(sandbox)

	def _init_model(self):
		if self._model_func is None:
			raise ValueError("You must save_model first")
		if self._model is None:
			import types
			import marshal
			import cPickle

			code = marshal.loads(self._model_func)
			sandbox = cPickle.loads(self._sandbox)
			globals()[self._api] = __import__(self._api)
			for k, v in sandbox.iteritems():
				globals()[k] = v

			func = types.FunctionType(code, globals(), "create_model")
			self._model = func()

			if self._api == 'lasagne':
				import lasagne
				layers = lasagne.layers.get_all_layers(self._model)
				i = [l.input_var for l in layers if isinstance(l, lasagne.layers.InputLayer)]
				self._pred = theano.function(
                                    inputs=i,
                                    outputs=lasagne.layers.get_output(self._model),
                                    allow_input_downcast=True,
                                    on_unused_input=None)
			else:
				raise NotImplementedError()

	def pred(self, *X):
		self._init_model()
		return self._pred(*X)

	# ==================== History manager ==================== #
	def clear(self):
		self._history = []

	def record(self, values, *tags):
		# in GMT
		curr_time = int(round(time.time() * 1000)) # in ms

		if not isinstance(tags, list) and not isinstance(tags, tuple):
			tags = [tags]
		tags = set(tags)

		# timestamp must never equal
		if len(self._history) > 0 and self._history[-1][0] >= curr_time:
			curr_time = self._history[-1][0] + 1

		self._history.append([curr_time, tags, values])

	def update(self, tags, func, after=None, before=None, n=None, absolute=False):
		''' Apply a funciton to all selected value

		Parameters
		----------
		tags : list, str, filter function or any comparable object
			get all values contain given tags
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
			row[2] = func(row[2])
			count += 1

	def select(self, tags, after=None, before=None, n=None,
		filter_value=None, absolute=False,
		newest_first=False, return_time=False):
		''' Query in history

		Parameters
		----------
		tags : list, str, filter function or any comparable object
			get all values contain given tags
		after, before : time constraint (in millisecond)
			after < t < before
		n : int
			number of record return
		filter_value : function
			function to filter each value found
		absolute : boolean
			whether required the same set of tags or just contain
		newest_first : boolean
			returning order
		return_time : boolean
			whether return time tags

		Returns
		-------
		return : list
			list of all values found
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

		if newest_first:
			return list(reversed(res))
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

	# ====== early stop ====== #
	def earlystop(self, tags, generalization_lost = False, generalization_sensitive=False, hope_hop=False):
		values = self.select(tags)
		values = [np.mean(i) if hasattr(i, '__len__') else i for i in values]
		shouldSave = 0
		shouldStop = 0
		if generalization_lost:
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

	# ==================== Load & Save ==================== #
	def save(self, path=None):
		if path is None and self._save_path is None:
			raise ValueError("Save path haven't specified!")
		path = path if path is not None else self._save_path
		self._save_path = path

		import cPickle
		from array import array

		f = h5py.File(path, 'w')
		f['history'] = cPickle.dumps(self._history)

		b = array("B", self._model_func)
		f['model_func'] = cPickle.dumps(b)
		f['api'] = self._api
		f['sandbox'] = self._sandbox

		for i, w in enumerate(self._weights):
			f['weight_%d' % i] = w
		f['nb_weights'] = len(self._weights)
		f.close()

	@staticmethod
	def load(path):
		if not os.path.exists(path):
			m = Model()
			m._save_path = path
			return m
		import cPickle

		m = Model()
		m._save_path = path

		f = h5py.File(path, 'r')
		m._history = cPickle.loads(f['history'].value)

		b = cPickle.loads(f['model_func'].value)
		m._model_func = b.tostring()
		m._api = f['api'].value
		m._sandbox = f['sandbox'].value

		m._weights = []
		for i in xrange(f['nb_weights'].value):
			m._weights.append(f['weight_%d' % i].value)

		f.close()
		return m

class Trainer(object):

	"""docstring for Trainer"""

	def __init__(self, arg):
		super(Trainer, self).__init__()
		self.arg = arg

# ======================================================================
# Data Preprocessing
# ======================================================================
class _batch(object):

	"""docstring for _batch"""

	def __init__(self, dataset, name):
		super(_batch, self).__init__()
		if (dataset is None or name is None):
			raise AttributeError('Must specify (hdf,name) or data')
		self._dataset = dataset
		self._name = name

		self._data = None
		if self._name in self._dataset.hdf:
			self._data = dataset.hdf[name]

	# ==================== Properties ==================== #
	@property
	def shape(self):
		return self._data.shape

	@property
	def dtype(self):
		return self._data.dtype

	@property
	def value(self):
		return self._data.value

	# ==================== Arithmetic ==================== #
	def sum2(self, axis=0):
		''' sum(X^2) '''
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
		s = self.sum(axis)
		return s / self.shape[axis]

	def var(self, axis=0):
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

	# ==================== Safty first ==================== #

	def _append_data(self, data):
		curr_size = self._data.shape[0]
		self._data.resize(curr_size + data.shape[0], 0)
		self._data[curr_size:] = data

	def _check_data(self, shape, dtype):
		if self._data is None:
			if self._name not in self._dataset:
				self._dataset.hdf.create_dataset(self._name, dtype=dtype,
					shape=(0,) + shape[1:], maxshape=(None, ) + shape[1:], chunks=True)
			self._data = self._dataset.hdf[self._name]

		if self._data.shape[1:] != shape[1:]:
			raise TypeError('Shapes not match ' + str(self.shape) + ' - ' + str(shape))

	# ==================== manupilation ==================== #
	def __add__(self, other):
		if not isinstance(other, np.ndarray):
			raise TypeError('Addition only for numpy ndarray')
		self._check_data(other.shape, other.dtype)
		self._append_data(other)
		return self

	def __mul__(self, other):
		if not isinstance(other, int):
			raise TypeError('Only mutilply with int')
		if self._data is None:
			raise TypeError("Data haven't initlized yet")
		copy = self._data[:]
		for i in xrange(other - 1):
			self._append_data(copy)
		return self

	def __len__(self):
		return self._data.shape[0]

	def iter(self, shuffle=True, batch_size=None):
		block_batch = self._dataset.create_batch(self.shape[0], batch_size=batch_size)
		for block, idx_batches in block_batch.iteritems():
			data = self._data[block[0]:block[1] + 1]
			batches = idx_batches[1]
			# shuffle is performed on big block, increase reliablity and performance
			if shuffle:
				idx = idx_batches[0]
				data = data[idx]
			# return smaller batches
			for b in batches:
				s = b[0] - block[0]
				e = b[1] - block[0] + 1
				if self._dataset.normalizer is not None:
					yield self._dataset.normalizer(data[s:e])
				else:
					yield data[s:e]

	def __iter__(self):
		block_batch = self._dataset.create_batch(self.shape[0])
		for block, idx_batches in block_batch.iteritems():
			data = self._data[block[0]:block[1] + 1]
			batches = idx_batches[1]
			# shuffle is performed on big block, increase reliablity and performance
			if self._dataset.shuffle:
				idx = idx_batches[0]
				data = data[idx]
			# return smaller batches
			for b in batches:
				s = b[0] - block[0]
				e = b[1] - block[0] + 1
				if self._dataset.normalizer is not None:
					yield self._dataset.normalizer(data[s:e])
				else:
					yield data[s:e]

	def __getitem__(self, key):
		if self._dataset.normalizer is not None:
		    return self._dataset.normalizer(self._data[key])
		else:
		    return self._data[key]

	def __setitem__(self, key, value):
		self._check_data(value.shape, value.dtype)
		if isinstance(key, slice):
			self._data[key] = value

	def __str__(self):
		if self._data is None:
			return 'None'
		return '<' + self._name + ' ' + str(self.shape) + ' ' + str(self.dtype) + '>'

class Dataset(object):

	'''
		Example
		-------
			def normalization(x):
				return x.astype(np.float32)

			d = dnntoolkit.Dataset('tmp.hdf', 'w', normalizer=normalization, batch_size=2, shuffle=True)
			d['X'] = np.zeros((2, 3))
			print('X' in d)
			>>> True

			d['X'][:1] = np.ones((1, 3))
			print(d['X'][:])
			>>> [[1,1,1],
			>>>	 [0,0,0]]

			for i in xrange(2):
				d['X'] += np.ones((1, 3))
			print(d['X'][:])
			>>> [[1,1,1],
			>>>	 [0,0,0],
			>>>  [1,1,1],
			>>>  [1,1,1]]

			d['X'] *= 2 # duplicate the data
			print(d['X'][:])
			>>> [[1,1,1],
			>>>	 [0,0,0],
			>>>  [1,1,1],
			>>>  [1,1,1],
			>>>  [1,1,1],
			>>>	 [0,0,0],
			>>>  [1,1,1],
			>>>  [1,1,1]]

			for i in d['X']: # shuffle configuration inherit from dataset
				print(str(i.shape) + '-' + str(i.dtype))
			>>> (2,3) - float64
			>>> (2,3) - float64
			>>> (2,3) - float64
			>>> (2,3) - float64

			d.shuffle_data() # for new order of data
			for i in d['X'].iter(shuffle=True):
				print(str(i.shape) + '-' + str(i.dtype))

			d.close()
	'''

	def __init__(self, path, mode='r', batch_size=512, normalizer=None, shuffle=True):
		super(Dataset, self).__init__()
		if not isinstance(batch_size, int):
			raise TypeError('Batch size must be integer')

		self.hdf = h5py.File(path, mode=mode)
		self._datamap = {}
		self.batch_size = batch_size
		self.shuffle = shuffle # shuffle is done on each block

		self.normalizer = normalizer
		self._seed = 12082518

		self._start = 0
		self._end = -1

	def bound(self, start=0, end=-1):
		self._start = start
		self._end = end

	# ==================== Arithmetic ==================== #
	def all_dataset(self, fileter_func=None, path='/'):
		res = []
		for p in self.hdf[path].keys():
			p = os.path.join(path, p)
			if 'Dataset' in str(type(self.hdf[p])):
				if fileter_func is not None and not fileter_func(p):
					continue
				res.append(p)
			elif 'Group' in str(type(self.hdf[p])):
				res += self.all_dataset(fileter_func, path=p)
		return res

	# ==================== Main ==================== #
	def shuffle_data(self):
		''' re-shuffle dataset to make sure new order come every epoch '''
		import time
		self._seed = int(time.time())

	def create_batch(self, nb_samples, batch_size = None):
		start = self._start
		end = self._end

		if batch_size is None:
			batch_size = self.batch_size
		block_size = 8 * batch_size # load big block, then yield smaller batches
		# makesure all iterator give same order
		np.random.seed(self._seed)

		idx = np.arange(0, nb_samples) # this will consume a lot memory
		# ceil is safer for GPU, expected worst case of smaller number of batch size
		n_block = max(int(np.ceil(nb_samples / block_size)), 1)
		block_jobs = mpi.segment_job(idx, n_block)
		batch_jobs = []
		for j in block_jobs:
			n_batch = max(int(np.ceil(len(j) / batch_size)), 1)
			job = mpi.segment_job(j, n_batch)
			batch_jobs.append([(i[0], i[-1]) for i in job])
		jobs = OrderedDict()
		for i, j in izip(block_jobs, batch_jobs):
			idx = np.random.permutation(len(i))
			jobs[(i[0], i[-1])] = (idx, j)
		return jobs

	def __getitem__(self, key):
		if key not in self._datamap:
			self._datamap[key] = _batch(self, key)
		return self._datamap[key]

	def __setitem__(self, key, value):
		if key not in self.hdf:
			self.hdf.create_dataset(key, data=value, dtype=value.dtype,
				maxshape=(None,) + value.shape[1:], chunks=True)
		if key not in self._datamap:
			self._datamap[key] = _batch(self, key)

	def __contains__(self, key):
		return key in self.hdf

	def close(self):
		self.hdf.close()

	def __del__(self):
		try:
			self.hdf.close()
			del self.hdf
		except:
			pass
# ======================================================================
# Visualiztion
# ======================================================================
class Visual():
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
	def print_hinton(arr, max_arr=None):
		''' Print hinton diagrams in terminal for visualization of weights
		Example:
			W = np.random.rand(10,10)
			hinton_print(W)
		'''
		def visual(val, max_val):
		    if abs(val) == max_val:
		        step = len(Visual.chars) - 1
		    else:
		        step = int(abs(float(val) / max_val) * len(Visual.chars))
		    colourstart = ""
		    colourend = ""
		    if val < 0:
		        colourstart, colourend = '\033[90m', '\033[0m'
		    return colourstart + Visual.chars[step] + colourend

		if max_arr is None:
		    max_arr = arr
		max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))
		# print(np.array2string(arr,
		#                       formatter={'float_kind': lambda x: visual(x, max_val)},
		#                       max_line_width=5000)
		# )
		f = np.vectorize(visual)
		result = f(arr, max_val)
		for r in result:
			print(''.join(r))
		return result

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

class Logger():
	chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

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
		if Logger._last_time < 0:
			Logger._last_time = time.time()
		eta = (max_val - p) / max(1e-13, abs(p - Logger._last_value)) * (time.time() - Logger._last_time)
		etd = time.time() - Logger._last_time
		Logger._last_value = p
		Logger._last_time = time.time()
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

		fh = logging.FileHandler(logging_path, mode='w')
		fh.setFormatter(logging.Formatter(
			fmt = '%(asctime)s %(levelname)s  %(message)s',
			datefmt = '%d/%m/%Y %I:%M:%S'))
		fh.setLevel(logging.DEBUG)

		sh = logging.StreamHandler(sys.stderr)
		sh.setFormatter(logging.Formatter(
			fmt = '%(asctime)s %(levelname)s  %(message)s',
			datefmt = '%d/%m/%Y %I:%M:%S'))
		sh.setLevel(logging.DEBUG)

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
class Speech():

    """docstring for Speech"""

    def __init__(self):
    	super(Speech, self).__init__()

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
        	signal = Speech.preprocess(signal)

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
        	signal = Speech.preprocess(signal)

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
		for ytrue, ypred in izip(y_true, y_pred):
			results.append(Speech.LevenshteinDistance(ytrue, ypred) / len(ytrue))
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
