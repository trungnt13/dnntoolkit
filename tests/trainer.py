from __future__ import print_function, division

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import theano
from theano import tensor

import numpy as np
import scipy as sp

import dnntoolkit
import lasagne
from matplotlib import pyplot as plt
np.random.seed(dnntoolkit.MAGIC_SEED)
# ======================================================================
# Global
# ======================================================================
W_saved = None
W_rollbacked = None
dnntoolkit.logger.set_save_path('tmp/log.txt')

# ======================================================================
#  data
# ======================================================================
ds = dnntoolkit.dataset.load_mnist()
dnntoolkit.logger.log(ds)

# ======================================================================
# Model
# ======================================================================
def ffnet(indim, outdim):
    indim = int(indim)
    outdim = int(outdim)
    l_in = lasagne.layers.InputLayer(shape=(None, indim))
    l_in = lasagne.layers.DropoutLayer(l_in, p=0.3)
    l_hid = lasagne.layers.DenseLayer(l_in, num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_hid = lasagne.layers.DropoutLayer(l_hid, p=0.3)
    return lasagne.layers.DenseLayer(l_hid, num_units=outdim,
        nonlinearity=lasagne.nonlinearities.softmax)

# ======================================================================
# Load data
# ======================================================================
m = dnntoolkit.model('tmp/tmp.ai')
m.set_model(ffnet, api='lasagne',
            indim=np.prod(ds['X_train'].shape[1:]),
            outdim=ds['y_train'].shape[1])
net = m.create_model()

y = tensor.matrix(name='y', dtype=theano.config.floatX)
input_var = [l.input_var for l in lasagne.layers.find_layers(net, types=lasagne.layers.InputLayer)]
dropout = lasagne.layers.find_layers(net, types=lasagne.layers.DropoutLayer)

# ====== Create prediction ====== #
y_pred_deter = lasagne.layers.get_output(net, deterministic=True)
f_pred = theano.function(
    inputs=input_var,
    outputs=y_pred_deter,
    allow_input_downcast=True
)
dnntoolkit.logger.info('Built prediction function!')

# ====== Create accuracy ====== #
cost_monitor = lasagne.objectives.categorical_accuracy(y_pred_deter, y).mean()
f_cost = theano.function(
    inputs=input_var + [y],
    outputs=cost_monitor,
    allow_input_downcast=True
)
dnntoolkit.logger.info('Built cost function!')

# ====== Create training ====== #
y_pred_stoch = lasagne.layers.get_output(net, deterministic=False)
cost_train = lasagne.objectives.categorical_crossentropy(y_pred_stoch, y).mean()

params = lasagne.layers.get_all_params(net)
lr = dnntoolkit.tensor.shared_scalar(name='lr', val=0.001)
updates = lasagne.updates.sgd(cost_train, params, lr)
f_updates = theano.function(
    inputs=input_var + [y],
    outputs=cost_train,
    updates=updates,
    allow_input_downcast=True
)
dnntoolkit.logger.info('Built updates function!')

# ======================================================================
# Train
# ======================================================================
trainer = dnntoolkit.trainer()
trainer.set_dataset(ds,
    train=['X_train', 'y_train'],
    valid=['X_valid', 'y_valid'],
    test=['X_test', 'y_test'])
trainer.set_model(cost_func=f_cost, updates_func=f_updates)
trainer.set_strategy(
    task='train',
    data={'train': ['X_train', 'y_train'], 'valid': ['X_valid', 'y_valid']},
    epoch=50,
    batch=256,
    shuffle=True,
    validfreq=0.6,
    seed=dnntoolkit.MAGIC_SEED
).set_strategy(
    task='test',
    batch=256
)

# ==================== Callback ==================== #
def batch_start(trainer):
    X = trainer.data[0]
    y = trainer.data[1]
    trainer.data = (X.reshape(-1, X.shape[1] * X.shape[2]), y)

def epoch_end(trainer):
    m.record(np.mean(trainer.cost), trainer.task, 'epoch_end')
    # ====== Visual weights ====== #
    weights = m.get_weights()
    nrows = int(np.ceil(np.sqrt(len(weights))))
    ncols = nrows
    fig = plt.figure()
    for i, w in enumerate(weights[:-1]):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        dnntoolkit.visual.plot_weights(w, ax)
    ax = fig.add_subplot(nrows, ncols, i + 2)
    dnntoolkit.visual.plot_weights(weights[-1], ax, colorbar='all',
        path='img/W_%d.png' % trainer.epoch)

def batch_end(trainer):
    m.record(np.mean(trainer.cost), trainer.task, 'batch_end')

def valid_end(trainer):
    m.record(np.mean(trainer.cost), 'valid_end')
    cost = [1 - i for i in m.select('valid_end')]
    shouldSave, shoudlStop = dnntoolkit.earlystop(cost, generalization_loss=True, threshold=3)
    if shouldSave:
        # dnntoolkit.logger.info('\nShould save!')
        m.save()
        global W_saved
        W_saved = [i.astype(np.float32) for i in m.get_weights()]
    if shoudlStop:
        # dnntoolkit.logger.info('\nShould stop!')
        trainer.stop()

def train_end(trainer):
    m.record(np.mean(trainer.cost), 'train_end')

def test_start(trainer):
    m.rollback() # rollback to best saved version of AI
    global W_rollbacked
    W_rollbacked = [i.astype(np.float32) for i in m.get_weights()]

def test_end(trainer):
    m.record(np.mean(trainer.cost), 'test_end')

trainer.set_callback(batch_start=batch_start,
                    epoch_end=epoch_end, batch_end=batch_end,
                    train_end=train_end, valid_end=valid_end,
                    test_start=test_start, test_end=test_end)

# ==================== Start now ==================== #
print(trainer)
trainer.run()

# trainer.set_strategy(yaml='tmp.yaml')
# print(trainer)
# trainer.run()

# ======================================================================
# Test load model
# ======================================================================
m = dnntoolkit.model.load('tmp/tmp.ai')
net = m.create_model()
W = m.get_weights()
dnntoolkit.logger.critical('******* Compare to best saved weights: ********')
for i, j in zip(W, W_saved):
    dnntoolkit.logger.critical('W differences: %.4f' % (np.sum(i - j)))
if W_rollbacked is not None:
    dnntoolkit.logger.critical('******* Compare to rollbacked weights: ********')
    for i, j in zip(W, W_rollbacked):
        dnntoolkit.logger.critical('W differences: %.4f' % (np.sum(i - j)))

# ====== Test prediction ====== #
test_pred = m.pred(ds['X_test'][:].reshape(-1, 28 * 28))
test_pred = np.argmax(test_pred, axis=1)
test_true = np.argmax(ds['y_test'][:], axis=1)
hit = np.sum(test_pred == test_true)
dnntoolkit.logger.critical('Test accuracy: %.4f' % (hit / len(test_true)))

# ====== Some training information ====== #
dnntoolkit.logger.info('Epoch cost:')
dnntoolkit.visual.print_bar(m.select(['epoch_end', 'train']), bincount=50)

dnntoolkit.logger.info('Validation accuracy:')
dnntoolkit.visual.print_bar(m.select(['valid_end']), bincount=50)

# ======================================================================
# End
# ======================================================================
ds.close()
