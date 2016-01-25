from __future__ import print_function, division

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
from theano import tensor

import numpy as np
import scipy as sp

import dnntoolkit
import unittest
import h5py
import lasagne
# ===========================================================================
# Model
# ===========================================================================
def test():
    l_in = lasagne.layers.InputLayer(shape=(None, 28, 28))
    l_in = lasagne.layers.FlattenLayer(l_in)
    l_in = lasagne.layers.DropoutLayer(l_in, p=0.3)

    l_hid = lasagne.layers.DenseLayer(l_in, num_units=128)
    l_hid = lasagne.layers.DropoutLayer(l_hid, p=0.3)

    l_out = lasagne.layers.DenseLayer(l_hid, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

# ===========================================================================
# Test
# ===========================================================================
ds = dnntoolkit.dataset.load_mnist()
print(ds)

m = dnntoolkit.model('/volumes/backup/model/mnist.ai')
m.set_model(test, 'lasagne')
ai = m.create_model()
input_var = [i.input_var
for i in lasagne.layers.find_layers(ai, types=lasagne.layers.InputLayer)]

y = tensor.matrix('y')
y_pred_stoch = lasagne.layers.get_output(ai, deterministic=False)
y_pred_deter = lasagne.layers.get_output(ai, deterministic=True)

cost_monitor = lasagne.objectives.categorical_accuracy(y_pred_deter, y).mean()
cost_train = lasagne.objectives.categorical_crossentropy(y_pred_stoch, y).mean()
regu = lasagne.regularization.L1L2(l2=dnntoolkit.dnn.calc_weights_decay(m.get_nweights()))
cost_regu = regu.apply_network(ai)
cost_train += cost_regu

params = lasagne.layers.get_all_params(ai, trainable=True)
grad = tensor.grad(cost_train, params)
print('Suggest learning rate: %f' % dnntoolkit.dnn.calc_lr(m.get_nweights(), m.get_nlayers()))
lr = dnntoolkit.tensor.shared_scalar(
    dnntoolkit.dnn.calc_lr(m.get_nweights(), m.get_nlayers()))
updates = lasagne.updates.rmsprop(grad, params,
    learning_rate=lr)

f_cost = theano.function(
    inputs=input_var + [y],
    outputs=cost_monitor,
    allow_input_downcast=True)
f_update = theano.function(
    inputs=input_var + [y],
    outputs=cost_train,
    updates=updates,
    allow_input_downcast=True)

# ===========================================================================
# Callback
# ===========================================================================
 # => valid Stats: Mean:0.9354 Var:0.00 Med:0.94 Min:0.91 Max:0.97
stopAllow = 3
def validend(trainer):
    m.record(np.mean(trainer.cost), 'validend')
    cost = [1. - i for i in m.select('validend')]
    save, stop = dnntoolkit.dnn.earlystop(cost, generalization_loss=True, threshold=3)
    if save:
        print('\nSaving !!!')
        m.save()
    if stop:
        global stopAllow
        if stopAllow > 0:
            print('\nDecreasing lr !!!')
            lr.set_value(lr.get_value() / 2)
            # m.rollback()
        else:
            print('\nStopping !!!')
            trainer.stop()
        stopAllow -= 1


# ===========================================================================
# TRainer
# ===========================================================================
Xcross = np.random.rand(1000, 28, 28)
ycross = dnntoolkit.tensor.to_categorical(np.random.randint(0, 10, size=1000))

def cross_it(*args):
    # print('**cross_it**')
    idx = range(1000)
    for i, j in zip(idx, idx[1:]):
        yield Xcross[i:j], ycross[i:j]

def Xcross_it(*args):
    # print('**IT**')
    idx = range(1000)
    for i, j in zip(idx, idx[1:]):
        yield Xcross[i:j]
def ycross_it(*args):
    idx = range(1000)
    for i, j in zip(idx, idx[1:]):
        yield ycross[i:j]

def Xcross_it_new(size, shuffle, seed, mode):
    # print('IT new')
    np.random.seed(seed)
    batches = dnntoolkit.mpi.segment_job(range(1000), int(1000 / size))
    np.random.shuffle(batches)
    for i in batches:
        yield Xcross[i[0]:i[-1]]
def ycross_it_new(size, shuffle, seed, mode):
    np.random.seed(seed)
    batches = dnntoolkit.mpi.segment_job(range(1000), int(1000 / size))
    np.random.shuffle(batches)
    for i in batches:
        yield ycross[i[0]:i[-1]]

trainer = dnntoolkit.trainer()
trainer.set_callback(valid_end=validend)
trainer.set_dataset(ds,
    valid=['X_valid', ds['y_valid'].value],
    test=['X_test', 'y_test'],
    cross=[cross_it],
    pcross=0.1
)
trainer.set_model(f_cost, f_update)
trainer.set_strategy(
    task='train',
    epoch=100,
    batch=128,
    validfreq=0.6,
    shuffle=True,
    data=[ds['X_train'].value, ds['y_train']],
    seed=12082518,
    # cross=[Xcross_it, ycross_it],
    pcross=0.2).set_strategy(
    task='test',
    batch=128)
print(trainer)
trainer.run()
