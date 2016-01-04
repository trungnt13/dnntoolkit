from __future__ import print_function, division

import numpy as np

import theano
from theano import tensor

import dnntoolkit
import lasagne
import models
reload(models)

np.random.seed(dnntoolkit.MAGIC_SEED)
# ======================================================================
# Simulate data
# ======================================================================
n = 1024
dim = 13
k = 5

X_train = np.random.rand(n, dim)
y_train = dnntoolkit.tensor.to_categorical(np.random.randint(0, k, size=n), n_classes=k)

X_valid = np.random.rand(n, dim)
y_valid = dnntoolkit.tensor.to_categorical(np.random.randint(0, k, size=n), n_classes=k)

X_test = np.random.rand(n * 2, dim)
y_test = dnntoolkit.tensor.to_categorical(np.random.randint(0, k, size=n * 2), n_classes=k)

ds = dnntoolkit.dataset('tmp.hdf', mode='w')
ds['X_train'].append(X_train)
ds['y_train'].append(y_train)
ds['X_valid'].append(X_valid)
ds['y_valid'].append(y_valid)
ds['X_test'].append(X_test)
ds['y_test'].append(y_test)

ds.close()

X_train = np.random.rand(n * 2, dim)
y_train = dnntoolkit.tensor.to_categorical(np.random.randint(0, k, size=n * 2), n_classes=k)

X_valid = np.random.rand(n, dim)
y_valid = dnntoolkit.tensor.to_categorical(np.random.randint(0, k, size=n), n_classes=k)

X_test = np.random.rand(n * 3, dim)
y_test = dnntoolkit.tensor.to_categorical(np.random.randint(0, k, size=n * 3), n_classes=k)

ds = dnntoolkit.dataset('tmp1.hdf', mode='w')
ds['X_train'].append(X_train)
ds['y_train'].append(y_train)
ds['X_valid'].append(X_valid)
ds['y_valid'].append(y_valid)
ds['X_test'].append(X_test)
ds['y_test'].append(y_test)

ds.close()

# ======================================================================
# Load data
# ======================================================================
ds = dnntoolkit.dataset('tmp.hdf', mode='r')

m = dnntoolkit.model.load('tmp.ai')
m.reset_history()
m.set_model(models.ffnet, api='lasagne', dim=dim, k=k, hid=30)
net = m.create_model()

y = tensor.matrix(name='y', dtype=theano.config.floatX)
input_var = [l.input_var for l in lasagne.layers.find_layers(net, names='input')]
y_pred = lasagne.layers.get_output(net)
f_pred = theano.function(
    inputs=input_var,
    outputs=y_pred,
    allow_input_downcast=True
)

cost = lasagne.objectives.categorical_crossentropy(y_pred, y).mean()
cost_monitor = lasagne.objectives.categorical_accuracy(y_pred, y)
f_cost = theano.function(
    inputs=input_var + [y],
    outputs=cost_monitor,
    allow_input_downcast=True
)

params = lasagne.layers.get_all_params(net)
lr = dnntoolkit.tensor.shared_scalar(name='lr', val=0.01)
updates = lasagne.updates.sgd(cost, params, lr)
f_updates = theano.function(
    inputs=input_var + [y],
    outputs=cost,
    updates=updates,
    allow_input_downcast=True
)

# ======================================================================
# Train
# ======================================================================
trainer = dnntoolkit.trainer()
trainer.set_dataset(ds, test=['X_test', 'y_test'])
trainer.set_model(pred_func=f_pred, cost_func=f_cost, updates_func=f_updates)
trainer.set_strategy(
    task='train',
    data={'train': ['X_train', 'y_train'], 'valid': ['X_valid', 'y_valid']},
    epoch=3,
    batch=50,
    shuffle=True,
    validfreq=25,
    seed=dnntoolkit.MAGIC_SEED
).set_strategy(
    task='valid',
    data=['X_valid', 'y_valid'],
    batch=100
).set_strategy(
    task='test',
    data=['X_test', 'y_test'],
    batch=80
)

def epoch_start(trainer):
    print('epoch start: task:%s epoch:%d' % (trainer.task, trainer.epoch))
    pass
def epoch_end(trainer):
    print('epoch end: task:%s' % (trainer.task))
    pass
def batch_start(trainer):
    m.record(None, trainer.task, 'batch_start')
    # print('batch start')
    pass
def batch_end(trainer):
    m.record(np.mean(trainer.cost), trainer.task, trainer.idx)
    # print('batch end')
    pass
def valid_end(trainer):
    print('valid end: ')
    print(len(trainer.cost))
    pass
def train_end(trainer):
    print('train end: %d-epoch %d-iter' % (trainer.epoch, trainer.iter))
    print(len(trainer.cost))
    pass
def test_end(trainer):
    print('test end:')
    print(len(trainer.cost))
    pass
trainer.set_callback(epoch_start=epoch_start, epoch_end=epoch_end,
        batch_start=batch_start, batch_end=batch_end,
        train_end=train_end, valid_end=valid_end, test_end=test_end)
print(trainer)
trainer.run()

# trainer.set_strategy(yaml='tmp.yaml')
# print(trainer)
# trainer.run()
# ======================================================================
# End
# ======================================================================
ds.close()
m.set_weights(lasagne.layers.get_all_param_values(net))
m.save('tmp.ai')
