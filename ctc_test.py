import numpy as np
import pickle
import theano
import theano.tensor as T
import lasagne

import ctc_cost

# ======================================================================
# Generate data
# * X_mask, y_pred_mask: decide what sample in input sequence is kept to procedure
# the output sequence
# * y_mask: decide what labels in output_sequence are matter for the training process.
# ======================================================================
nb_samples = 30 # n
nb_classes = 3	# C
in_seq_len = 8 # T
out_seq_len = 5 # L
nb_features = 7

X = np.random.rand(nb_samples, in_seq_len, nb_features)
X_mask = np.ones(shape=(nb_samples, in_seq_len))

y_pred = np.random.rand(nb_samples, in_seq_len, nb_classes + 1) # include 1 blank
y_pred_mask = np.ones(shape=(nb_samples, in_seq_len))
y_pred_mask[0, 1] = y_pred_mask[0, 1] - 1 # remove 2nd sample in first input sequence

y = np.random.randint(0, nb_classes,
	size =(nb_samples, out_seq_len))
y_mask = np.ones(shape=(nb_samples, out_seq_len))

for i in xrange(y_pred.shape[0]):
	y_pred[i,:,:] = y_pred[i,:,:] / np.sum(y_pred[i,:,:], axis=-1).reshape(-1, 1)

# X = theano.shared(X)
# X_mask = theano.shared(X_mask)
# y_pred = theano.shared(y_pred)
# y_pred_mask = theano.shared(y_pred_mask)

# y = theano.shared(y)
# y_mask = theano.shared(y_mask)

print('Samples:%d' % nb_samples)
print('Classes:%d' % nb_classes)
print('InSeq:%d' % in_seq_len)
print('OutSeq:%d' % out_seq_len)
print('Features:%d' % nb_features)

# ======================================================================
# Test the model
# ======================================================================
print("Building model ...")

X_ = T.tensor3('X', dtype='float32')           # B x T x F   # only matrix because i use embedding...
X_mask_ = T.matrix('X_mask', dtype='float32')  # B x T
y_ = T.matrix('y', dtype='float32')            # B x L
y_mask_ = T.matrix('y_mask', dtype='float32')  # B x L

l_inp = lasagne.layers.InputLayer((None, in_seq_len, nb_features)) # [nb_samples,in_seq_len,nb_features]
l_rec = lasagne.layers.LSTMLayer(l_inp, num_units=100) # [nb_samples,in_seq_len, 100]
l_shp = lasagne.layers.reshape(l_rec, (nb_samples * in_seq_len, 100))  # [nb_samples, 100]
l_ctc = lasagne.layers.DenseLayer(l_shp, nb_classes + 1,
								nonlinearity=lasagne.nonlinearities.linear) # include blank: 11200x(4+1)
l_out = lasagne.layers.reshape(l_ctc, (nb_samples, in_seq_len, nb_classes + 1)) # 16x700x(4+1)

# ====== Variable ====== #
y_pred = lasagne.layers.get_output(l_out, X_)
all_params = lasagne.layers.get_all_params(l_out, trainable=True)
grad_cost, real_cost = ctc_cost.ctc_objective(y_pred, y_)
all_grads = T.grad(grad_cost, all_params)
updates = lasagne.updates.rmsprop(all_grads, all_params, learning_rate=1e-1)

# ====== Funcition ====== #
train = theano.function(inputs=[X_, y_],
                        outputs=[grad_cost, real_cost],
                        updates=updates,
                        allow_input_downcast=True,
                        on_unused_input='ignore')
y_pred_softmax = T.exp(y_pred) / T.exp(y_pred).sum(axis=-1)[:,:, None]
pred = theano.function(inputs=[X_], outputs=y_pred_softmax, allow_input_downcast=True)
grad_obj = theano.function(inputs=[X_, y_], outputs=grad_cost, allow_input_downcast=True)
real_obj = theano.function(inputs=[X_, y_], outputs=real_cost, allow_input_downcast=True)
grad_check = theano.function(inputs=[X_, y_], outputs=all_grads, allow_input_downcast=True)

# ====== Training ====== #
for i in xrange(1000):
	if i % 10 == 0:
		print('Epoch %d: ' % i + str(train(X, y)))
	for t in grad_check(X, y):
		if np.isnan(np.min(t)) or np.isinf(np.sum(t)):
			print('Gradient exploded!')

print('Prediction:')
print(np.argmax(pred(X), -1))
print(y)
