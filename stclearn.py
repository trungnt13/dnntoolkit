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

def sgd(learning_rate):
	def sgd_func(loss, params):
		grads = theano.grad(loss, params)
		updates = OrderedDict()
		for param, grad in zip(params, grads):
			updates[param] = param - learning_rate * grad
		return updates
	return sgd_func

def sgd_momentum(learning_rate, momentum=0.9):
	def momentum(loss, params):
		grads = theano.grad(loss, params)
		updates = OrderedDict()
		for param, grad in zip(params, grads):
			value = param.get_value(borrow=True)
			velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			x = momentum * velocity + (param - learning_rate * grad)
			updates[velocity] = x - param
			updates[param] = x
		return updates
	return momentum

def sgd_nesterov_momentum(learning_rate, momentum=0.9):
	def momentum(loss, params):
		grads = theano.grad(loss, params)
		updates = OrderedDict()
		for param, grad in zip(params, grads):
			value = param.get_value(borrow=True)
			velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
			x = momentum * velocity + (learning_rate * grad)
			updates[velocity] = x
			updates[param] = momentum * x + updates[param]
		return updates
	return momentum

def adagrad(learning_rate=1.0, epsilon=1e-6):
	def adagrad_func(loss, params):
		grads = theano.grad(loss, params)
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

def rmsprop(learning_rate=1.0, rho=0.9, epsilon=1e-6):
	def rmsprop_func(loss, params):
		grads = theano.grad(loss, params)
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

def adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-6):
	def adadelta_func(loss, params):
		grads = theano.grad(loss, params)
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


def adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
	def adam_func(loss, params):
		all_grads = theano.grad(loss, params)
		t_prev = theano.shared(utils.floatX(0.))
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


def norm_constraint(tensor_var, max_norm, norm_axes=None, epsilon=1e-7):
    """Max weight norm constraints and gradient clipping

    This takes a TensorVariable and rescales it so that incoming weight
    norms are below a specified constraint value. Vectors violating the
    constraint are rescaled so that they are within the allowed range.

    Parameters
    ----------
    tensor_var : TensorVariable
        Theano expression for update, gradient, or other quantity.
    max_norm : scalar
        This value sets the maximum allowed value of any norm in
        `tensor_var`.
    norm_axes : sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `tensor_var`. When this is not specified and `tensor_var` is a
        matrix (2D), this is set to `(0,)`. If `tensor_var` is a 3D, 4D or
        5D tensor, it is set to a tuple listing all axes but axis 0. The
        former default is useful for working with dense layers, the latter
        is useful for 1D, 2D and 3D convolutional layers.
        (Optional)
    epsilon : scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.

    Returns
    -------
    TensorVariable
        Input `tensor_var` with rescaling applied to weight vectors
        that violate the specified constraints.

    Examples
    --------
    >>> param = theano.shared(
    ...     np.random.randn(100, 200).astype(theano.config.floatX))
    >>> update = param + 100
    >>> update = norm_constraint(update, 10)
    >>> func = theano.function([], [], updates=[(param, update)])
    >>> # Apply constrained update
    >>> _ = func()
    >>> from lasagne.utils import compute_norms
    >>> norms = compute_norms(param.get_value())
    >>> np.isclose(np.max(norms), 10)
    True

    Notes
    -----
    When `norm_axes` is not specified, the axes over which the norm is
    computed depend on the dimensionality of the input variable. If it is
    2D, it is assumed to come from a dense layer, and the norm is computed
    over axis 0. If it is 3D, 4D or 5D, it is assumed to come from a
    convolutional layer and the norm is computed over all trailing axes
    beyond axis 0. For other uses, you should explicitly specify the axes
    over which to compute the norm using `norm_axes`.
    """
    ndim = tensor_var.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(ndim)
        )

    dtype = np.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    target_norms = T.clip(norms, 0, dtype(max_norm))
    constrained_output = \
        (tensor_var * (target_norms / (dtype(epsilon) + norms)))

    return constrained_output


def total_norm_constraint(tensor_vars, max_norm, epsilon=1e-7,
                          return_norm=False):
    """Rescales a list of tensors based on their combined norm

    If the combined norm of the input tensors exceeds the threshold then all
    tensors are rescaled such that the combined norm is equal to the threshold.

    Scaling the norms of the gradients is often used when training recurrent
    neural networks [1]_.

    Parameters
    ----------
    tensor_vars : List of TensorVariables.
        Tensors to be rescaled.
    max_norm : float
        Threshold value for total norm.
    epsilon : scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.
    return_norm : bool
        If true the total norm is also returned.

    Returns
    -------
    tensor_vars_scaled : list of TensorVariables
        The scaled tensor variables.
    norm : Theano scalar
        The combined norms of the input variables prior to rescaling,
        only returned if ``return_norms=True``.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> import lasagne
    >>> from lasagne.updates import sgd, total_norm_constraint
    >>> x = T.matrix()
    >>> y = T.ivector()
    >>> l_in = InputLayer((5, 10))
    >>> l1 = DenseLayer(l_in, num_units=7, nonlinearity=T.nnet.softmax)
    >>> output = lasagne.layers.get_output(l1, x)
    >>> cost = T.mean(T.nnet.categorical_crossentropy(output, y))
    >>> all_params = lasagne.layers.get_all_params(l1)
    >>> all_grads = T.grad(cost, all_params)
    >>> scaled_grads = total_norm_constraint(all_grads, 5)
    >>> updates = sgd(scaled_grads, all_params, learning_rate=0.1)

    Notes
    -----
    The total norm can be used to monitor training.

    References
    ----------
    .. [1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014): Sequence to sequence
       learning with neural networks. In Advances in Neural Information
       Processing Systems (pp. 3104-3112).
    """
    norm = T.sqrt(sum(T.sum(tensor**2) for tensor in tensor_vars))
    dtype = np.dtype(theano.config.floatX).type
    target_norm = T.clip(norm, 0, dtype(max_norm))
    multiplier = target_norm / (dtype(epsilon) + norm)
    tensor_vars_scaled = [step * multiplier for step in tensor_vars]

    if return_norm:
        return tensor_vars_scaled, norm
    else:
        return tensor_vars_scaled

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

	# ====== scan ====== #
	def step(i, X, y, X_mask, y_mask):
		y_pred = lasagne.layers.get_output(layer, _get_input_and_mask_layer(layers, X, X_mask))
		cost = cost_func(y_pred, y, X_mask, y_mask, **args)
		update = updater(cost, all_params)
		update[_cost] = T.set_subtensor(_cost[i:], T.cast(cost, floatX))
		return update

	result, update = theano.scan(step,
		sequences=[_cost_idx, X_, y_, X_mask_, y_mask_])

	s = theano.function(inputs=[X_, y_, X_mask_, y_mask_],
		updates=update,
		allow_input_downcast=True)

	def trainer(X, y, X_mask=None, y_mask=None):
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
	return trainer
