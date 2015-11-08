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
# ======================================================================
from __future__ import print_function, division

import os
import sys

import numpy as np
import scipy as sp

import theano
from theano import tensor as T

import h5py

from itertools import izip

import sidekit
import soundfile

import paramiko
from stat import S_ISDIR

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
class Dataset():

	"""docstring for HDF5Dataset"""

	def __init__(self, path):
		super(Dataset, self).__init__()
		self.hdf = h5py.File(path, 'w')

	def put(self, data, append=True):
		pass

	@property
	def shape(self):
	    return self.dataset.shape

	def __len__(self):
		return self.dataset.shape[0]

	def __getitem__(self, key):
		if isinstance(key, slice):
		    if key.stop + self.start <= self.end:
		        idx = slice(key.start + self.start, key.stop + self.start)
		    else:
		        raise IndexError
		elif isinstance(key, int):
		    if key + self.start < self.end:
		        idx = key + self.start
		    else:
		        raise IndexError
		elif isinstance(key, np.ndarray):
		    if np.max(key) + self.start < self.end:
		        idx = (self.start + key).tolist()
		    else:
		        raise IndexError
		elif isinstance(key, list):
		    if max(key) + self.start < self.end:
		        idx = [x + self.start for x in key]
		    else:
		        raise IndexError
		if self.normalizer is not None:
		    return self.normalizer(self.data[idx])
		else:
		    return self.data[idx]


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
	def logmel(signal, fs, n_filters=40, nwin=256, shift=0.01, delta=True, normalize=True):
		if len(signal.shape) > 1:
			signal = signal.ravel()

		#####################################
		# 1. Some const.
		# n_filters = 40 # The number of mel filter bands
		n_ceps = 13 # The number of cepstral coefficients
		f_min = 0. # The minimal frequency of the filter bank

		# overlap = nwin - int(shift * fs)

		#####################################
		# 2. preprocess.
		signal = Speech.preprocess(signal)

		#####################################
		# 3. logmel.
		# MFCC
		logmel = sidekit.frontend.features.mfcc(signal,
						lowfreq=f_min, maxfreq=fs / 2,
						nlinfilt=0, nlogfilt=n_filters, nfft=512,
						fs=fs, nceps=n_ceps, midfreq=1000,
						nwin=nwin, shift=shift,
						get_spec=False, get_mspec=True)
		# energy = logmel[1]
		logmel = logmel[3]

		#####################################
		# 4. delta.
		if delta:
			delta1 = sidekit.frontend.features.compute_delta(logmel,
							win=3, method='filter')
			delta2 = sidekit.frontend.features.compute_delta(delta1,
							win=3, method='filter')
			logmel = np.concatenate((logmel, delta1, delta2), 1)

		# VAD
		idx = sidekit.frontend.vad.vad_snr(signal, 30, fs=fs, shift=shift, nwin=nwin)
		logmel = logmel[idx, :]

		# Normalize
		if normalize:
			mean = np.mean(logmel, axis = 0)
			var = np.var(logmel, axis = 0)
			logmel = (logmel - mean) / np.sqrt(var)

		return logmel

	@staticmethod
	def mfcc(signal, fs, n_ceps, delta=True, normalize=True):
		#####################################
		# 1. Const.
		fs = 8000
		n_filters = 40 # The number of filter bands
		f_min = 0. # The minimal frequency of the filter bank

		nwin = 256
		shift = 0.01

		#####################################
		# 2. Speech.
		signal = Speech.preprocess(signal)

		#####################################
		# 3. mfcc.
		# MFCC
		mfcc = sidekit.frontend.features.mfcc(signal,
						lowfreq=f_min, maxfreq=fs / 2,
						nlinfilt=0, nlogfilt=n_filters, nfft=512,
						fs=fs, nceps=n_ceps, midfreq=1000,
						nwin=nwin, shift=shift,
						get_spec=False, get_mspec=False)[0]
		delta1 = sidekit.frontend.features.compute_delta(mfcc,
						win=3, method='filter')
		delta2 = sidekit.frontend.features.compute_delta(delta1,
						win=3, method='filter')
		mfcc = np.concatenate((mfcc, delta1, delta2), 1)

		# VAD
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
