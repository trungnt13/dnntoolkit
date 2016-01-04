from __future__ import print_function, division

import theano
from theano import tensor

import dnntoolkit
import lasagne

def ffnet(dim, k, hid=50):
    l_in = lasagne.layers.InputLayer(name='input', shape=(None, dim))
    l_hidden = lasagne.layers.DenseLayer(l_in, num_units=hid,
        nonlinearity=lasagne.nonlinearities.rectify, name='hidden')
    l_out = lasagne.layers.DenseLayer(l_hidden, num_units=k,
        nonlinearity=lasagne.nonlinearities.softmax, name='out')
    return l_out
