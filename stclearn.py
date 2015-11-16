"""
Implementation of accelerated stochastic learning (weight updates after every training example).
This version still very slow compare to batch learning, however, it is 2 times faster than the
normal way of feeding data one-by-one in batch

Examples
--------
>>> import lasagne
>>> import theano.tensor as T
>>> import theano
>>> from lasagne.nonlinearities import softmax
>>> from lasagne.layers import InputLayer, DenseLayer, get_output
>>> from lasagne.updates import sgd, apply_momentum
>>> l_in = InputLayer((100, 20))
>>> l1 = DenseLayer(l_in, num_units=3, nonlinearity=softmax)
>>> x = T.matrix('x')  # shp: num_batch x num_features
>>> y = T.ivector('y') # shp: num_batch
>>> l_out = get_output(l1, x)
>>> params = lasagne.layers.get_all_params(l1)
>>> loss = T.mean(T.nnet.categorical_crossentropy(l_out, y))
>>> updates_sgd = sgd(loss, params, learning_rate=0.0001)
>>> updates = apply_momentum(updates_sgd, params, momentum=0.9)
>>> train_function = theano.function([x, y], updates=updates)
"""

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

import lasagne

MAX_BATCH_SIZE = 1024

floatX = theano.config.floatX
# ====== For record the cost ====== #
_cost = theano.shared(np.zeros(MAX_BATCH_SIZE).astype(floatX)) # stupid fix
_cost_idx = theano.shared(np.arange(MAX_BATCH_SIZE).astype(np.int32))

def sgd(learning_rate, max_norm=5):
	def sgd_func(loss, params):
		grads = theano.grad(loss, params)
		if max_norm:
			grads = lasagne.updates.total_norm_constraint(grads, max_norm)

		updates = OrderedDict()
		for param, grad in zip(params, grads):
			updates[param] = param - learning_rate * grad
		return updates
	return sgd_func

def sgd_momentum(learning_rate, momentum=0.9, max_norm=5):
	def momentum_func(loss, params):
		grads = theano.grad(loss, params)
		if max_norm:
			grads = lasagne.updates.total_norm_constraint(grads, max_norm)

		updates = OrderedDict()
		for param, grad in zip(params, grads):
			value = param.get_value(borrow=True)
			velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			x = momentum * velocity + (param - learning_rate * grad)
			updates[velocity] = x - param
			updates[param] = x
		return updates
	return momentum_func

def sgd_nesterov_momentum(learning_rate, momentum=0.9, max_norm=5):
	def momentum_func(loss, params):
		grads = theano.grad(loss, params)
		if max_norm:
			grads = lasagne.updates.total_norm_constraint(grads, max_norm)

		updates = OrderedDict()
		for param, grad in zip(params, grads):
			value = param.get_value(borrow=True)
			velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			x = momentum * velocity + (learning_rate * grad)
			updates[velocity] = x
			updates[param] = momentum * x + updates[param]
		return updates
	return momentum_func

def adagrad(learning_rate=1.0, epsilon=1e-6, max_norm=5):
	def adagrad_func(loss, params):
		grads = theano.grad(loss, params)
		if max_norm:
			grads = lasagne.updates.total_norm_constraint(grads, max_norm)

		updates = OrderedDict()
		for param, grad in zip(params, grads):
		    value = param.get_value(borrow=True)
		    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
		                         broadcastable=param.broadcastable)
		    accu_new = accu + grad ** 2
		    updates[accu] = accu_new
		    updates[param] = param - (learning_rate * grad /
		                              T.sqrt(accu_new + epsilon))
		return updates
	return adagrad_func

def rmsprop(learning_rate=1.0, rho=0.9, epsilon=1e-6, max_norm=5):
	def rmsprop_func(loss, params):
		grads = theano.grad(loss, params)
		if max_norm:
			grads = lasagne.updates.total_norm_constraint(grads, max_norm)

		updates = OrderedDict()
		for param, grad in zip(params, grads):
		    value = param.get_value(borrow=True)
		    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
		                         broadcastable=param.broadcastable)
		    accu_new = rho * accu + (1 - rho) * grad ** 2
		    updates[accu] = accu_new
		    updates[param] = param - (learning_rate * grad /
		                              T.sqrt(accu_new + epsilon))
		    return updates
	return rmsprop_func

def adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-6, max_norm=5):
	def adadelta_func(loss, params):
		grads = theano.grad(loss, params)
		if max_norm:
			grads = lasagne.updates.total_norm_constraint(grads, max_norm)

		updates = OrderedDict()
		for param, grad in zip(params, grads):
			value = param.get_value(borrow=True)
			# accu: accumulate gradient magnitudes
			accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
			                     broadcastable=param.broadcastable)
			# delta_accu: accumulate update magnitudes (recursively!)
			delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
			                       broadcastable=param.broadcastable)
			# update accu (as in rmsprop)
			accu_new = rho * accu + (1 - rho) * grad ** 2
			updates[accu] = accu_new
			# compute parameter update, using the 'old' delta_accu
			update = (grad * T.sqrt(delta_accu + epsilon) /
			          T.sqrt(accu_new + epsilon))
			updates[param] = param - learning_rate * update
			# update delta_accu (as accu, but accumulating updates)
			delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
			updates[delta_accu] = delta_accu_new
			return updates
	return adadelta_func


def adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_norm=5):
	def adam_func(loss, params):
		all_grads = theano.grad(loss, params)
		if max_norm:
			all_grads = lasagne.updates.total_norm_constraint(all_grads, max_norm)

		t_prev = theano.shared(lasagne.utils.floatX(0.))
		updates = OrderedDict()
		t = t_prev + 1
		a_t = learning_rate * T.sqrt(1 - beta2**t) / (1 - beta1**t)
		for param, g_t in zip(params, all_grads):
			value = param.get_value(borrow=True)
			m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
			                       broadcastable=param.broadcastable)
			v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
			                       broadcastable=param.broadcastable)

			m_t = beta1 * m_prev + (1 - beta1) * g_t
			v_t = beta2 * v_prev + (1 - beta2) * g_t**2
			step = a_t * m_t / (T.sqrt(v_t) + epsilon)

			updates[m_prev] = m_t
			updates[v_prev] = v_t
			updates[param] = param - step
		updates[t_prev] = t
		return updates
	return adam_func


def _get_input_and_mask_layer(layers, X, X_mask):
	inputs = {}
	for i, l in enumerate(layers):
		if isinstance(l, lasagne.layers.InputLayer):
			if len(l.shape) == 2:
				inputs[l] = X_mask
			elif len(l.shape) == 3:
				inputs[l] = X
	return inputs

def get_trainer(layer, cost_func, updater=sgd(1e-4), **args):

	# ====== Input variable ====== #
	all_params = lasagne.layers.get_all_params(layer, trainable=True)
	layers = lasagne.layers.get_all_layers(layer)

	X_ = T.tensor4(dtype=floatX)
	y_ = T.tensor3(dtype=floatX)
	X_mask_ = T.tensor3(dtype=floatX)
	y_mask_ = T.tensor3(dtype=floatX)

	# ====== train ====== #
	def step_update(i, X, y, X_mask, y_mask):
		y_pred = lasagne.layers.get_output(layer, _get_input_and_mask_layer(layers, X, X_mask))
		cost = cost_func(y_pred, y, X_mask, y_mask, **args)
		update = updater(cost, all_params)
		update[_cost] = T.set_subtensor(_cost[i:], T.cast(cost, floatX))
		return update

	result, update = theano.scan(step_update,
		sequences=[_cost_idx, X_, y_, X_mask_, y_mask_])

	s = theano.function(inputs=[X_, y_, X_mask_, y_mask_],
		updates=update,
		allow_input_downcast=True)

	# ====== cost ====== #
	def step_cost(X, y, X_mask, y_mask):
		y_pred = lasagne.layers.get_output(layer, _get_input_and_mask_layer(layers, X, X_mask))
		cost = cost_func(y_pred, y, X_mask, y_mask, **args)
		return cost

	result, update = theano.scan(step_cost,
		sequences=[X_, y_, X_mask_, y_mask_],
		outputs_info=None)

	c = theano.function(inputs=[X_, y_, X_mask_, y_mask_],
		outputs=result,
		allow_input_downcast=True)

	# ====== create function ====== #
	def train_function(X, y, X_mask=None, y_mask=None):
		if X_mask is None:
			X_mask = np.ones(X.shape[:-1])
		if y_mask is None:
			y_mask = np.ones(y.shape)
		# ====== expand 1 more dimension to represent batch_size=1 ====== #
		# resize to 4D
		X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
		# resize to 3D
		X_mask = X_mask.reshape((X_mask.shape[0], 1, X_mask.shape[1]))

		# resize to 3D
		y = y.reshape((y.shape[0], 1, y.shape[1]))
		# resize to 3D
		y_mask = y_mask.reshape((y_mask.shape[0], 1, y_mask.shape[1]))

		# run training function
		s(X, y, X_mask, y_mask)

		return _cost[:(X.shape[0] - 1)].eval()

	def cost_function(X, y, X_mask=None, y_mask=None):
		if X_mask is None:
			X_mask = np.ones(X.shape[:-1])
		if y_mask is None:
			y_mask = np.ones(y.shape)
		# ====== expand 1 more dimension to represent batch_size=1 ====== #
		# resize to 4D
		X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
		# resize to 3D
		X_mask = X_mask.reshape((X_mask.shape[0], 1, X_mask.shape[1]))

		# resize to 3D
		y = y.reshape((y.shape[0], 1, y.shape[1]))
		# resize to 3D
		y_mask = y_mask.reshape((y_mask.shape[0], 1, y_mask.shape[1]))

		return c(X, y, X_mask, y_mask)

	return train_function, cost_function
