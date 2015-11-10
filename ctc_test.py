import numpy as np
import theano
import theano.tensor as T
import lasagne
import time

# ======================================================================
# Generate data
# * X_mask, y_pred_mask: decide what sample in input sequence is kept to procedure
# the output sequence
# * y_mask: decide what labels in output_sequence are matter for the training process.
# ======================================================================
nb_samples = 1000 # n
nb_classes = 20	# C
in_seq_len = 60 # T
out_seq_len = 20 # L
nb_features = 80

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
def ctc_test():
	import ctc_cost
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
	updates = lasagne.updates.rmsprop(all_grads, all_params, learning_rate=1e-3)

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
	start_time = time.time()
	for i in xrange(100):
		grad_cost, real_cost = train(X, y)
		if i % 10 == 0:
			print('Epoch %d: ' % i + str(grad_cost) + '-' + str(real_cost))
		for t in grad_check(X, y):
			if np.isnan(np.min(t)) or np.isinf(np.sum(t)):
				print('Gradient exploded!')
	print('Training time %.2f seconds' % (time.time() - start_time))

	print('Prediction:')
	print(np.argmax(pred(X), -1))
	print(y)

def ctc_test1():
	import ctc_cost
	import stclearn

	X_mask = np.ones(X.shape[:-1])

	# ====== Build model ====== #
	linp = lasagne.layers.InputLayer(shape=(None, in_seq_len, nb_features)) # [in_seq_len, nb_features]
	lmask = lasagne.layers.InputLayer(shape=(None, in_seq_len))
	lstm = lasagne.layers.LSTMLayer(linp, num_units=100, mask_input=lmask)
	lreshape1 = lasagne.layers.reshape(lstm, shape=(-1, 100))
	linear = lasagne.layers.DenseLayer(lreshape1, nb_classes + 1,
		nonlinearity=lasagne.nonlinearities.linear)
	lreshape2 = lasagne.layers.reshape(linear, shape=(-1, in_seq_len, nb_classes + 1))

	# ====== Trainer ====== #
	print('Building trainer ...')
	train = stclearn.get_trainer(lreshape2, ctc_cost.ctc_objective,
		updater=stclearn.sgd(1e-3), batch=False)

	# ====== Training ====== #
	print('Training ...')
	for i in xrange(100):
		start_time = time.time()
		cost = train(X, y, X_mask)
		print('Epoch %d: ' % i + str(np.mean(cost)))
		print('Training time %.2f seconds' % (time.time() - start_time))

# ctc_test()
ctc_test1()
