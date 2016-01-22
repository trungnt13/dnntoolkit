from __future__ import print_function, division

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
from theano import tensor

import numpy as np
import scipy as sp

import lasagne
import dnntoolkit
import cPickle
import time
import h5py

from itertools import izip
from collections import OrderedDict

