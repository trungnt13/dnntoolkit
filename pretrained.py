from __future__ import print_function

import numpy as np

import theano
from theano import tensor

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.layers import Conv1DLayer, MaxPool1DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import LocalResponseNormalization2DLayer

# ======================================================================
# Utils
# ======================================================================
def create_dropout(l, name=None, p=0.5):
    p = theano.shared(np.cast[theano.config.floatX](p), name=name)
    return lasagne.layers.DropoutLayer(l, p=p, name=name)

def create_gaussian_noise(l, name=None, sigma=0.075):
    sigma = theano.shared(np.cast[theano.config.floatX](sigma), name=name)
    return lasagne.layers.GaussianNoiseLayer(l, sigma=sigma, name=name)

# ======================================================================
# Vision
# ======================================================================

def vggnet(params_path, api='lasagne'):
    ''' Pretrained VGGnet from:
        - https://github.com/Lasagne/Recipes/blob/master/examples/ImageNet%20Pretrained%20Network%20(VGG_S).ipynb
        - Input: images of size (3,224,224)
        - Output: 1000 categories

    Parameters
    ----------
    params_path : str
        path to pretrained model file (https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl)

    Returns
    -------
    model : object
        - lasagne: output layer of the network
        - blocks: b
        - keras: a
    '''
    #####################################
    # 1. Create model.
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = Conv2DDNNLayer(net['input'], num_filters=96, filter_size=7, stride=2)
    net['norm1'] = LocalResponseNormalization2DLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = MaxPool2DLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)

    net['conv2'] = Conv2DDNNLayer(net['pool1'], num_filters=256, filter_size=5)
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)

    net['conv3'] = Conv2DDNNLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)

    net['conv4'] = Conv2DDNNLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)

    net['conv5'] = Conv2DDNNLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)

    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)

    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)

    # output
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = net['fc8']

    #####################################
    # 2. load params.

    import pickle

    model = pickle.load(open(params_path))
    # CLASSES = model['synset words']
    # MEAN_IMAGE = model['mean image']

    lasagne.layers.set_all_param_values(output_layer, model['values'])

    return output_layer

def alexnet(params_path, api='lasagne'):
    pass

# ======================================================================
# Speech
# ======================================================================
def deep_wide(dim, nout, arch):
    '''
    Parameters
    ----------
    arch : int, str
        [p1]    arch=1 => [dense,4000,sigmoid] (extremely wide)
        [p2+p3] arch=2 => 7*[dense,2048,rectify,dropout=0.3]
        [p5]    arch=3 => 6*[dense,1024,sigmoid]
    Notes
    -----
    [p1] Suggested features by the paper: 51 frames x 15 band energies
    [p2+p3] Suggested 15 input frames, 25-ms window, 10-ms fixed frame rate.
    12th-order MFCCs & energy, along with their first and second derivatives.
    [p4] Suggested Static 40-log-filter-banks only (19-frames)
    References
    ----------
     - p1: Deep and Wide: Multiple Layers in Automatic Speech Recognition
     - p2: Deep Neural Networks for Acoustic Modeling in Speech Recognition
     - p3: Acoustic Modeling using Deep Belief Networks
     - p4: RECENT ADVANCES IN DEEP LEARNING FOR SPEECH RESEARCH AT MICROSOFT
     - p5: Kaldi recipe
    '''
    l_in = InputLayer(shape=(None, dim), name=['input', 'pretrain'])
    if arch == 1:
        l_hid = DenseLayer(l_in, num_units=10000,
            nonlinearity=lasagne.nonlinearities.sigmoid)
        l_hid = DenseLayer(l_in, num_units=10000,
            nonlinearity=lasagne.nonlinearities.sigmoid)
    elif arch == 2:
        l_hid = DenseLayer(l_in, num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            name='pretrain')
        l_hid = create_dropout(l_hid, name=['pretrain', 'dropout'], 0.3)
        l_hid = DenseLayer(l_hid, num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            name='pretrain')
        l_hid = create_dropout(l_hid, name=['pretrain', 'dropout'], 0.3)
        l_hid = DenseLayer(l_hid, num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            name='pretrain')
        l_hid = create_dropout(l_hid, name=['pretrain', 'dropout'], 0.3)
        l_hid = DenseLayer(l_hid, num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            name='pretrain')
        l_hid = create_dropout(l_hid, name=['pretrain', 'dropout'], 0.3)
        l_hid = DenseLayer(l_hid, num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            name='pretrain')
        l_hid = create_dropout(l_hid, name=['pretrain', 'dropout'], 0.3)
        l_hid = DenseLayer(l_hid, num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            name='pretrain')
        l_hid = create_dropout(l_hid, name=['pretrain', 'dropout'], 0.3)
        l_hid = DenseLayer(l_hid, num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            name='pretrain')
        l_hid = create_dropout(l_hid, name=['pretrain', 'dropout'], 0.3)
    elif arch == 3:
        l_hid = lasagne.layers.DimshuffleLayer(l_in, pattern=(0, 'x', 1))
        l_hid = Conv1DLayer(l_hid, num_filters=256, filter_size=5, pad=0, stride=1)
        l_hid = MaxPool1DLayer(l_hid, pool_size=3, stride=3, ignore_border=False)
        l_hid = Conv1DLayer(l_hid, num_filters=256, filter_size=5, pad=0, stride=1)
        l_hid = MaxPool1DLayer(l_hid, pool_size=3, stride=3, ignore_border=False)
        l_hid = Conv1DLayer(l_hid, num_filters=256, filter_size=5, pad=0, stride=2)
        l_hid = lasagne.layers.FlattenLayer(l_hid)

        l_hid = DenseLayer(l_hid, num_units=1024,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            name='pretrain')
        l_hid = DenseLayer(l_hid, num_units=1024,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            name='pretrain')
        l_hid = DenseLayer(l_hid, num_units=1024,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            name='pretrain')
        l_hid = DenseLayer(l_hid, num_units=1024,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            name='pretrain')
        l_hid = DenseLayer(l_hid, num_units=1024,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            name='pretrain')
        l_hid = DenseLayer(l_hid, num_units=1024,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            name='pretrain')

    l_out = DenseLayer(l_hid, num_units=nout,
        nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def deepspeech1(seqlen, dim, nout):
    '''

    References
    ----------
     - Deep Speech: Scaling up end-to-end speech recognition
       (http://arxiv.org/abs/1412.5567)
     - Deep Speech 2: End-to-End Speech Recognition in English and Mandarin
       (http://arxiv.org/abs/1512.02595)
    '''
    pass

def deepspeech2(seqlen, dim, nout):
    pass

def essen(seqlen, dim, nout):
    '''

    References
    ----------
     - EESEN: End-to-End Speech Recognition using Deep RNN Models and WFST-based Decoding
       (http://arxiv.org/abs/1507.08240)
    '''
    pass
