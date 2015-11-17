# coding=utf-8
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
# ======================================================================
from __future__ import print_function, division

import os
import sys
import math

from itertools import izip
from collections import OrderedDict

import numpy as np
import scipy as sp

import theano
from theano import tensor as T

import h5py


import sidekit
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

def batch_size(data_shape, gpu_model=GPU.K40, performance=None, bandwidth=None, size=None):
	if gpu_model is not None:
		performance = gpu_model['performance']
		bandwidth = gpu_model['bandwidth']
		size = gpu_model['size']
	elif performance is None or bandwidth is None or size is None:
		raise Exception('You must specify all GPU parameters')

	performance = 10**12 * performance * 4 # bytes/second
	bandwidth = bandwidth * 10**9 # bytes
	size = 0.01 * size * 10**9 # bytes
	data_mem = np.prod(data_shape) * 4 # bytes

	# point is: maximize computation time, reduce transfer, < 0.01 GPU mem
	batch_cost = []
	print('computation  -   transfer   -   capacity')
	for i in xrange(1, 20):
		batch_size = 2**i * np.prod(data_shape[1:]) * 4 # bytes
		n_batch = data_mem / batch_size
		print('%.10f - %.10f - %.10f' % (n_batch * (batch_size / performance), n_batch * (batch_size / bandwidth), (batch_size / size)))
		cost = n_batch * \
			((batch_size / performance) / (batch_size / bandwidth)) / \
			(batch_size / size)
		batch_cost.append(cost)
	print(batch_cost)
	return 2**(np.argmin(batch_cost) + 1)

# ======================================================================
# Early stopping
# ======================================================================
class EarlyStop(object):

	""" Implemetation of earlystop based on voting from many different techniques
	Example:
		e = EarlyStop([train_loss, ...],[validation_lost, ...])
		e.enable(generalization_loss=True/False, hope_hop=True/False)
		shouldSave, shouldStop = e.update(train_lost,None)
		shouldSave, shouldStop = e.update(None,validation_loss)
		shouldSave, shouldStop = e.update(train_loss,validation_loss)

		if shouldSave:
			# save the model
		if shouldStop:
			# stop training
	Reference:
		Early Stopping but when. (1999). Early Stopping but when, 1–15.
	"""
 	__train = []
 	__validation = []
 	__methods = []
 	__valid_iter = 0
 	__train_iter = 0

	def __init__(self, train_record=None, validation_record=None):
		super(EarlyStop, self).__init__()
		self.__methods.append(self.__update_hope_and_hop)
		if isinstance(train_record, list):
			self.__train = train_record[:]
		if isinstance(validation_record, list):
			self.__validation = validation_record[:]

	def enable(self, generalization_loss = False, hope_hop = False):
		self.__methods = []
		if generalization_loss:
			self.__methods.append(self.__update_gl)
		if hope_hop:
			self.__methods.append(self.__update_hope_and_hop)

	def reset(self):
		self.__train = []
		self.__validation = []
		self.__train_iter = 0
		self.__valid_iter = 0
		self.__methods.append(self.__update_hope_and_hop)

	def update(self, train_loss, validation_loss):
		'''
		return: shouldSave, shouldStop
		'''
		#####################################
		# 1. Update records.
		isUpdateTrain = False
		isUpdateValidation = False
		if train_loss is not None:
			if isinstance(train_loss, list):
				self.__train += train_loss
			else:
				self.__train.append(train_loss)
			isUpdateTrain = True
			self.__train_iter += 1
		if validation_loss is not None:
			if isinstance(validation_loss, list):
				self.__validation += validation_loss
			else:
				self.__validation.append(validation_loss)
			isUpdateValidation = True
			self.__valid_iter += 1

		#####################################
		# 2. start calculation.
		result = []
		for m in self.__methods:
			result.append(m(isUpdateTrain, isUpdateValidation))
		result = np.sum(result, 0)

		return result[0] > 0, result[1] > 0

	def debug(self):
		s = '\n'
		s += 'Train: ' + str(['%.2f' % i for i in self.__train]) + '\n'
		s += 'Validation: ' + str(['%.2f' % i for i in self.__validation]) + '\n'
		s += 'ValidIter:%d \n' % self.__valid_iter
		s += 'TrainIter:%d \n' % self.__train_iter
		s += '=========== Hope and hop ===========\n'
		s += 'Patience:%.2f \n' % self.patience
		s += 'Increase:%.2f \n' % self.patience_increase
		if len(self.__validation) > 0:
			s += 'Best:%.2f \n' % min(self.__validation)
		s += '=========== Generalization Loss ===========\n'
		s += 'Threshold:%.2f \n' % self.__gl_exit_threshold
		if len(self.__validation) > 0:
			s += 'GL:%.2f \n' % (100 * (self.__validation[-1] / min(self.__validation) - 1))
		return s

	# ==================== Generalization Sensitive ==================== #
	def __update_gs(self, train_update, validation_update):
		if len(self.__validation) == 0:
			return 0, 0
		shouldStop = 0
		shouldSave = 0

		if validation_update:
			if self.__validation[-1] > min(self.__validation):
				shouldStop = 1
				shouldSave = -1
			else:
				shouldStop = -1
				shouldSave = 1

		return shouldSave, shouldStop

	# ==================== Generalization Loss ==================== #
	__gl_exit_threshold = 3

	def __update_gl(self, train_update, validation_update):
		if len(self.__validation) == 0:
			return 0, 0
		shouldStop = 0
		shouldSave = 0

		if validation_update:
			gl_t = 100 * (self.__validation[-1] / min(self.__validation) - 1)
			if gl_t == 0: # min = current_value
				shouldSave = 1
				shouldStop = -1
			elif gl_t > self.__gl_exit_threshold:
				shouldStop = 1
				shouldSave = -1

		return shouldSave, shouldStop

	# ==================== Hope and hop ==================== #
	# Not so strict rules
	patience = 2
	patience_increase = 0.5
 	improvement_threshold = 0.998

	def __update_hope_and_hop(self, train_update, validation_update):
		if len(self.__validation) == 0:
			return 0, 0
		shouldStop = 0
		shouldSave = 0

		if validation_update:
			# one more iteration
			i = self.__valid_iter

			if len(self.__validation) == 1: # cold start
				shouldSave = 1
				shouldStop = -1
			else: # warm up
				last_best_validation = min(self.__validation[:-1])
				# significant improvement
				if min(self.__validation) < last_best_validation * self.improvement_threshold:
					self.patience += i * self.patience_increase
					shouldSave = 1
					shouldStop = -1
				# punish
				else:
					# the more increase the faster we running out of patience
					rate = self.__validation[-1] / last_best_validation
					self.patience -= i * self.patience_increase * rate
					# if still little bit better, just save it
					if min(self.__validation) < last_best_validation:
						shouldSave = 1

		if self.patience <= 0:
			shouldStop = 1
			shouldSave = -1

		return shouldSave, shouldStop

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
	def sum(self, axis=0):
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
		v = 0
		isInit = False
		n = self.shape[axis]
		for X in self.iter(shuffle=False):
			X = X.astype(np.float64)
			if axis == 0:
				v += np.sum(np.power(X, 2), axis) - \
                                    1 / n * np.power(np.sum(X, axis), 2)
			else:
				if not isInit:
					v = [np.sum(np.power(X, 2), axis) - 1 / n * np.power(np.sum(X, axis), 2)]
					isInit = True
				else:
					v.append(np.sum(np.power(X, 2), axis) - 1 / n * np.power(np.sum(X, axis), 2))
		if isinstance(v, list):
			v = np.concatenate(v, axis=0)
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
	def hinton_print(arr, max_arr=None):
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
	def hinton_plot(matrix, max_weight=None, ax=None):
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

	"""docstring for Logger"""

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
	def logmel(signal, fs, n_filters=40, nfft=512, win=0.025, shift=0.01,
			delta1=True, delta2=True, energy=True,
			normalize=True, vad=True, clean=True):
		if len(signal.shape) > 1:
			signal = signal.ravel()

		#####################################
		# 1. Some const.
		# n_filters = 40 # The number of mel filter bands
		n_ceps = 13 # The number of cepstral coefficients
		f_min = 0. # The minimal frequency of the filter bank
		f_max = fs / 2
		nwin = fs * win
		# overlap = nwin - int(shift * fs)

		#####################################
		# 2. preprocess.
		if clean:
			signal = Speech.preprocess(signal)

		#####################################
		# 3. logmel.
		logmel = sidekit.frontend.features.mfcc(signal,
						lowfreq=f_min, maxfreq=f_max,
						nlinfilt=0, nlogfilt=n_filters, nfft=nfft,
						fs=fs, nceps=n_ceps, midfreq=1000,
						nwin=nwin, shift=shift,
						get_spec=False, get_mspec=True)
		logenergy = logmel[1]
		logmel = logmel[3]
		# TODO: check how to calculate energy delta
		if energy:
			logmel = np.concatenate((logmel, logenergy.reshape(-1, 1)), axis=1)

		#####################################
		# 4. delta.
		tmp = [logmel]
		if delta1 or delta2:
			d1 = sidekit.frontend.features.compute_delta(logmel,
							win=3, method='filter')
			d2 = sidekit.frontend.features.compute_delta(delta1,
							win=3, method='filter')
			if delta1: tmp.append(d1)
			if delta2: tmp.append(d2)
		logmel = np.concatenate(tmp, 1)

		#####################################
		# 5. VAD and normalize.
		if vad:
			idx = sidekit.frontend.vad.vad_snr(signal, 30, fs=fs, shift=shift, nwin=nwin)
			logmel = logmel[idx, :]

		# Normalize
		if normalize:
			mean = np.mean(logmel, axis = 0)
			var = np.var(logmel, axis = 0)
			logmel = (logmel - mean) / np.sqrt(var)

		return logmel

	@staticmethod
	def mfcc(signal, fs, n_ceps, n_filters=40, nfft=512, win=0.025, shift=0.01,
			delta1=True, delta2=True, energy=True,
			normalize=True, vad=True, clean=True):
		#####################################
		# 1. Const.
		f_min = 0. # The minimal frequency of the filter bank
		f_max = fs / 2
		nwin = fs * win

		#####################################
		# 2. Speech.
		if clean:
			signal = Speech.preprocess(signal)

		#####################################
		# 3. mfcc.
		# MFCC
		mfcc = sidekit.frontend.features.mfcc(signal,
						lowfreq=f_min, maxfreq=f_max,
						nlinfilt=0, nlogfilt=n_filters, nfft=nfft,
						fs=fs, nceps=n_ceps, midfreq=1000,
						nwin=nwin, shift=shift,
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
			idx = sidekit.frontend.vad.vad_snr(signal, 30, fs=fs, shift=shift, nwin=nwin)
			mfcc = mfcc[idx, :]

		# Normalize
		if normalize:
			mean = np.mean(mfcc, axis = 0)
			var = np.var(mfcc, axis = 0)
			mfcc = (mfcc - mean) / np.sqrt(var)

		return mfcc

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
