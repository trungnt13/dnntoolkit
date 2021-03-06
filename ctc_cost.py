"""
CTC-Connectionist Temporal Classification

Code provided by "Mohammad Pezeshki" and "Philemon Brakel"- May. 2015 -
Montreal Institute for Learning Algorithms

Referece: Graves, Alex, et al. "Connectionist temporal classification:
labelling unsegmented sequence data with recurrent neural networks."
Proceedings of the 23rd international conference on Machine learning.
ACM, 2006.

Credits: Shawn Tan, Rakesh Var, Philemon Brakel and Mohammad Pezeshki

This code is distributed without any warranty, express or implied.
"""
import theano
import numpy as np
from theano import tensor
from theano import tensor as T


floatX = theano.config.floatX

# ======================================================================
# cost objective function
# ======================================================================
def ctc_objective(y_pred, y, y_pred_mask=None, y_mask=None, batch=True):
	''' CTC objective.

	Parameters
	----------
	y_pred : [nb_samples, in_seq_len, nb_classes+1]
		softmax probabilities
	y : [nb_samples, out_seq_len]
		output sequences
	y_mask : [nb_samples, out_seq_len]
		mask decides which labels in y is included (0 for ignore, 1 for keep)
	y_pred_mask : [nb_samples, in_seq_len]
		mask decides which samples in input sequence are used
	batch : True/False
		if batching is not used, nb_samples=1
		Note: the implementation without batch support is more reliable

	Returns
	-------
	grad_cost : the cost you calculate gradient on
	actual_cost : the cost for monitoring model performance (*NOTE: do not calculate
		gradient on this cost)

	Note
	----
	According to @Richard Kurle:
		test error of 38% with 1 bidirectional LSTM layer or with a stack of 3,
		but I could not reproduce the results to those reported in Grave's paper.

		If you get blanks only, you probably have just bad hyperparameters or you
		did not wait enough epochs. At the beginnign of the training,
		only the cost decreases but you don't see yet any characters popping up.

		You will need gradient clipping to prevent exploding gradients as well.
	'''
	y_pred_mask = y_pred_mask if y_pred_mask is not None else T.ones((y_pred.shape[0], y_pred.shape[1]), dtype=floatX)
	y_mask = y_mask if y_mask is not None else T.ones(y.shape, dtype=floatX)
	if batch:
		# ====== reshape input ====== #
		y_pred = y_pred.dimshuffle(1, 0, 2)
		y_pred_mask = y_pred_mask.dimshuffle(1, 0)
		y = y.dimshuffle(1, 0)
		y_mask = y_mask.dimshuffle(1, 0)

		# ====== calculate cost ====== #
		grad_cost = _pseudo_cost(y, y_pred, y_mask, y_pred_mask, False)
		grad_cost = grad_cost.mean()
		monitor_cost = _cost(y, y_pred, y_mask, y_pred_mask, True)
		monitor_cost = monitor_cost.mean()

		return grad_cost, monitor_cost
	else:
		y = T.cast(y, dtype='int32')

		# batch_size=1 => just take [0] to reduce 1 dimension
		y_pred = y_pred[0]
		y_pred_mask = y_pred_mask[0]
		y = y[0]
		y_mask = y_mask[0]

		# after take, ndim=2 go up to 3, need to be reduced back to 2
		y_pred = T.take(y_pred, T.nonzero(y_pred_mask, return_matrix=True), axis=0)[0]
		y = T.take(y, T.nonzero(y_mask, return_matrix=True), axis=0).ravel()

		return _cost_no_batch(y_pred, y)

# ======================================================================
# Sequence Implementation (NO batch)
# Shawn Tan
# ======================================================================
def _interleave_blanks(Y):
    Y_ = T.alloc(-1, Y.shape[0] * 2 + 1)
    Y_ = T.set_subtensor(Y_[T.arange(Y.shape[0]) * 2 + 1], Y)
    return Y_

def _create_skip_idxs(Y):
    skip_idxs = T.arange((Y.shape[0] - 3) // 2) * 2 + 1
    non_repeats = T.neq(Y[skip_idxs], Y[skip_idxs + 2])
    return skip_idxs[non_repeats.nonzero()]

def _update_log_p(skip_idxs, zeros, active, log_p_curr, log_p_prev):
    active_skip_idxs = skip_idxs[(skip_idxs < active).nonzero()]
    active_next = T.cast(T.minimum(
        T.maximum(
            active + 1,
            T.max(T.concatenate([active_skip_idxs, [-1]])) + 2 + 1
        ),
        log_p_curr.shape[0]
    ), 'int32')

    common_factor = T.max(log_p_prev[:active])
    p_prev = T.exp(log_p_prev[:active] - common_factor)
    _p_prev = zeros[:active_next]
    # copy over
    _p_prev = T.set_subtensor(_p_prev[:active], p_prev)
    # previous transitions
    _p_prev = T.inc_subtensor(_p_prev[1:], _p_prev[:-1])
    # skip transitions
    _p_prev = T.inc_subtensor(_p_prev[active_skip_idxs + 2], p_prev[active_skip_idxs])
    updated_log_p_prev = T.log(_p_prev) + common_factor

    log_p_next = T.set_subtensor(
        zeros[:active_next],
        log_p_curr[:active_next] + updated_log_p_prev
    )
    return active_next, log_p_next

def _path_probs(predict, Y, alpha=1e-4):
    smoothed_predict = (1 - alpha) * predict[:, Y] + alpha * np.float32(1.) / Y.shape[0]
    L = T.log(smoothed_predict)
    zeros = T.zeros_like(L[0])
    base = T.set_subtensor(zeros[:1], np.float32(1))
    log_first = zeros

    f_skip_idxs = _create_skip_idxs(Y)
    b_skip_idxs = _create_skip_idxs(Y[::-1]) # there should be a shortcut to calculating this

    def step(log_f_curr, log_b_curr, f_active, log_f_prev, b_active, log_b_prev):
        f_active_next, log_f_next = _update_log_p(f_skip_idxs, zeros, f_active, log_f_curr, log_f_prev)
        b_active_next, log_b_next = _update_log_p(b_skip_idxs, zeros, b_active, log_b_curr, log_b_prev)
        return f_active_next, log_f_next, b_active_next, log_b_next
    [f_active, log_f_probs, b_active, log_b_probs], _ = theano.scan(
        step,
        sequences=[
            L,
            L[::-1, ::-1]
        ],
        outputs_info=[
            np.int32(1), log_first,
            np.int32(1), log_first,
        ]
    )
    idxs = T.arange(L.shape[1]).dimshuffle('x', 0)
    mask = (idxs < f_active.dimshuffle(0, 'x')) & (idxs < b_active.dimshuffle(0, 'x'))[::-1, ::-1]
    log_probs = log_f_probs + log_b_probs[::-1, ::-1] - L
    return log_probs, mask

def _cost_no_batch(predict, Y):
    log_probs, mask = _path_probs(predict, _interleave_blanks(Y))
    common_factor = T.max(log_probs)
    total_log_prob = T.log(T.sum(T.exp(log_probs - common_factor)[mask.nonzero()])) + common_factor
    return -total_log_prob

# ======================================================================
# Batch support implementation by:
# Philemon Brakel
# Mohammad Pezeshki
# ======================================================================

def _get_targets(y, log_y_hat, y_mask, y_hat_mask):
    '''
    Returns the target values according to the CTC cost with respect to y_hat.
    Note that this is part of the gradient with respect to the softmax output
    and not with respect to the input of the original softmax function.
    All computations are done in log scale
    '''
    num_classes = log_y_hat.shape[2] - 1
    blanked_y, blanked_y_mask = _add_blanks(
        y=y,
        blank_symbol=num_classes,
        y_mask=y_mask)

    log_alpha, log_beta = _log_forward_backward(blanked_y,
                                                log_y_hat, blanked_y_mask,
                                                y_hat_mask, num_classes)
    # explicitly not using a mask to prevent inf - inf
    y_prob = _class_batch_to_labeling_batch(blanked_y, log_y_hat,
                                            y_hat_mask=None)
    marginals = log_alpha + log_beta - y_prob
    max_marg = marginals.max(2)
    max_marg = T.switch(T.le(max_marg, -np.inf), 0, max_marg)
    log_Z = T.log(T.exp(marginals - max_marg[:,:, None]).sum(2))
    log_Z = log_Z + max_marg
    log_Z = T.switch(T.le(log_Z, -np.inf), 0, log_Z)
    targets = _labeling_batch_to_class_batch(blanked_y,
                                             T.exp(marginals -
                                                   log_Z[:,:, None]),
                                             num_classes + 1)
    return targets


def _pseudo_cost(y, y_hat, y_mask, y_hat_mask, skip_softmax=False):
    '''
    Training objective.
    Computes the marginal label probabilities and returns the
    cross entropy between this distribution and y_hat, ignoring the
    dependence of the two.
    This cost should have the same gradient but hopefully theano will
    use a more stable implementation of it.
    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    y_hat : tensor3 (T, B, C)
        class probabily distribution sequences, potentially in log domain
    y_mask : matrix (L, B)
        indicates which values of y to use
    y_hat_mask : matrix (T, B)
        indicates the lenghts of the sequences in y_hat
    skip_softmax : bool
        whether to interpret y_hat as probabilities or unnormalized energy
        values. The latter might be more numerically stable and efficient
        because it avoids the computation of the explicit cost and softmax
        gradients.
    '''
    if skip_softmax:
        y_hat_softmax = (T.exp(y_hat - y_hat.max(2)[:,:, None]) /
                         T.exp(y_hat -
                               y_hat.max(2)[:,:, None]).sum(2)[:,:, None])
        y_hat_safe = y_hat - y_hat.max(2)[:,:, None]
        log_y_hat_softmax = (y_hat_safe -
                             T.log(T.exp(y_hat_safe).sum(2))[:,:, None])
        targets = _get_targets(y, log_y_hat_softmax, y_mask, y_hat_mask)
    else:
        y_hat_softmax = y_hat
        targets = _get_targets(y, (T.log(y_hat) -
                                  T.log(y_hat.sum(2)[:,:, None])),
                              y_mask, y_hat_mask)

    mask = y_hat_mask[:,:, None]
    if skip_softmax:
        y_hat_grad = y_hat_softmax - targets
        return (y_hat * mask *
                theano.gradient.disconnected_grad(y_hat_grad)).sum(0).sum(1)
    return -T.sum(theano.gradient.disconnected_grad(targets) *
                  T.log(y_hat**mask), axis=0).sum(1)


def _sequence_log_likelihood(y, y_hat, y_mask, y_hat_mask, blank_symbol,
                            log_scale=True):
    '''
    Based on code from Shawn Tan.
    Credits to Kyle Kastner as well.
    '''
    y_hat_mask_len = tensor.sum(y_hat_mask, axis=0, dtype='int32')
    y_mask_len = tensor.sum(y_mask, axis=0, dtype='int32')

    if log_scale:
        log_probabs = _log_path_probabs(y, T.log(y_hat),
                                        y_mask, y_hat_mask,
                                        blank_symbol)
        batch_size = log_probabs.shape[1]
        log_labels_probab = _log_add(
            log_probabs[y_hat_mask_len - 1,
                        tensor.arange(batch_size),
                        y_mask_len - 1],
            log_probabs[y_hat_mask_len - 1,
                        tensor.arange(batch_size),
                        y_mask_len - 2])
    else:
        probabilities = _path_probabs(y, y_hat,
                                      y_mask, y_hat_mask,
                                      blank_symbol)
        batch_size = probabilities.shape[1]
        labels_probab = (probabilities[y_hat_mask_len - 1,
                                       tensor.arange(batch_size),
                                       y_mask_len - 1] +
                         probabilities[y_hat_mask_len - 1,
                                       tensor.arange(batch_size),
                                       y_mask_len - 2])
        log_labels_probab = tensor.log(labels_probab)
    return log_labels_probab


def _cost(y, y_hat, y_mask, y_hat_mask, log_scale=True):
    '''
    Training objective.
    Computes the CTC cost using just the forward computations.
    The difference between this function and the vanilla 'cost' function
    is that this function adds blanks first.
    Note: don't try to compute the gradient of this version of the cost!
    ----
    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    y_hat : tensor3 (T, B, C)
        class probabily distribution sequences
    y_mask : matrix (L, B)
        indicates which values of y to use
    y_hat_mask : matrix (T, B)
        indicates the lenghts of the sequences in y_hat
    log_scale : bool
        uses log domain computations if True

    '''
    num_classes = y_hat.shape[2] - 1
    blanked_y, blanked_y_mask = _add_blanks(
        y=y,
        blank_symbol=num_classes,
        y_mask=y_mask)
    final_cost = -_sequence_log_likelihood(blanked_y, y_hat,
                                          blanked_y_mask, y_hat_mask,
                                          num_classes,
                                          log_scale=log_scale)
    return final_cost


def _add_blanks(y, blank_symbol, y_mask=None):
    '''Add blanks to a matrix and updates mask
    Input shape: L x B
    Output shape: 2L+1 x B
    '''
    # for y
    y_extended = y.T.dimshuffle(0, 1, 'x')
    blanks = tensor.zeros_like(y_extended) + blank_symbol
    concat = tensor.concatenate([y_extended, blanks], axis=2)
    res = concat.reshape((concat.shape[0],
                          concat.shape[1] * concat.shape[2])).T
    begining_blanks = tensor.zeros((1, res.shape[1])) + blank_symbol
    blanked_y = tensor.concatenate([begining_blanks, res], axis=0)
    # for y_mask
    if y_mask is not None:
        y_mask_extended = y_mask.T.dimshuffle(0, 1, 'x')
        concat = tensor.concatenate([y_mask_extended,
                                     y_mask_extended], axis=2)
        res = concat.reshape((concat.shape[0],
                              concat.shape[1] * concat.shape[2])).T
        begining_blanks = tensor.ones((1, res.shape[1]), dtype=floatX)
        blanked_y_mask = tensor.concatenate([begining_blanks, res], axis=0)
    else:
        blanked_y_mask = None
    return blanked_y.astype('int32'), blanked_y_mask


def _class_batch_to_labeling_batch(y, y_hat, y_hat_mask=None):
    '''
    Convert (T, B, C) tensor into (T, B, L) tensor.
    Notes
    -----
    T: number of time steps
    B: batch size
    L: length of label sequence
    C: number of classes
    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    y_hat : tensor3 (T, B, C)
        class probabily distribution sequences
    y_hat_mask : matrix (T, B)
        indicates the lenghts of the sequences in y_hat
    Returns
    -------
    tensor3 (T, B, L):
        A tensor that contains the probabilities per time step of the
        labels that occur in the target sequence.
    '''
    if y_hat_mask is not None:
        y_hat = y_hat * y_hat_mask[:,:, None]
    batch_size = y_hat.shape[1]
    y_hat = y_hat.dimshuffle(0, 2, 1)
    res = y_hat[:, y.astype('int32'), T.arange(batch_size)]
    return res.dimshuffle(0, 2, 1)


def _recurrence_relation(y, y_mask, blank_symbol):
    '''
    Construct a permutation matrix and tensor for computing CTC transitions.
    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    y_mask : matrix (L, B)
        indicates which values of y to use
    blank_symbol: integer
        indicates the symbol that signifies a blank label.
    Returns
    -------
    matrix (L, L)
    tensor3 (L, L, B)
    '''
    n_y = y.shape[0]
    blanks = tensor.zeros((2, y.shape[1])) + blank_symbol
    ybb = tensor.concatenate((y, blanks), axis=0).T
    sec_diag = (tensor.neq(ybb[:, :-2], ybb[:, 2:]) *
                tensor.eq(ybb[:, 1:-1], blank_symbol) *
                y_mask.T)

    # r1: LxL
    # r2: LxL
    # r3: LxLxB
    eye2 = tensor.eye(n_y + 2)
    r2 = eye2[2:, 1:-1]  # tensor.eye(n_y, k=1)
    r3 = (eye2[2:, :-2].dimshuffle(0, 1, 'x') *
          sec_diag.dimshuffle(1, 'x', 0))

    return r2, r3


def _path_probabs(y, y_hat, y_mask, y_hat_mask, blank_symbol):
    pred_y = _class_batch_to_labeling_batch(y, y_hat, y_hat_mask)
    pred_y = pred_y.dimshuffle(0, 2, 1)
    n_labels = y.shape[0]

    r2, r3 = _recurrence_relation(y, y_mask, blank_symbol)

    def step(p_curr, p_prev):
        # instead of dot product, we * first
        # and then sum oven one dimension.
        # objective: T.dot((p_prev)BxL, LxLxB)
        # solusion: Lx1xB * LxLxB --> LxLxB --> (sumover)xLxB
        dotproduct = (p_prev + tensor.dot(p_prev, r2) +
                      (p_prev.dimshuffle(1, 'x', 0) * r3).sum(axis=0).T)
        return p_curr.T * dotproduct * y_mask.T  # B x L

    probabilities, _ = theano.scan(
        step,
        sequences=[pred_y],
        outputs_info=[tensor.eye(n_labels)[0] * tensor.ones(y.T.shape)])
    return probabilities


def _log_add(a, b):
    # TODO: move functions like this to utils
    max_ = tensor.maximum(a, b)
    result = (max_ + tensor.log1p(tensor.exp(a + b - 2 * max_)))
    return T.switch(T.isnan(result), max_, result)


def _log_dot_matrix(x, z):
    y = x[:,:, None] + z[None,:,:]
    y_max = y.max(axis=1)
    out = T.log(T.sum(T.exp(y - y_max[:, None,:]), axis=1)) + y_max
    return T.switch(T.isnan(out), -np.inf, out)


def _log_dot_tensor(x, z):
    log_dot = x.dimshuffle(1, 'x', 0) + z
    max_ = log_dot.max(axis=0)
    out = (T.log(T.sum(T.exp(log_dot - max_[None,:,:]), axis=0)) + max_)
    out = out.T
    return T.switch(T.isnan(out), -np.inf, out)


def _log_path_probabs(y, log_y_hat, y_mask, y_hat_mask, blank_symbol,
                     reverse=False):
    '''
    Uses dynamic programming to compute the path probabilities.
    Notes
    -----
    T: number of time steps
    B: batch size
    L: length of label sequence
    C: number of classes
    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    log_y_hat : tensor3 (T, B, C)
        log class probabily distribution sequences
    y_mask : matrix (L, B)
        indicates which values of y to use
    y_hat_mask : matrix (T, B)
        indicates the lenghts of the sequences in log_y_hat
    blank_symbol: integer
        indicates the symbol that signifies a blank label.
    Returns
    -------
    tensor3 (T, B, L):
        the log forward probabilities for each label at every time step.
        masked values should be -inf
    '''

    n_labels, batch_size = y.shape

    if reverse:
        y = y[::-1]
        log_y_hat = log_y_hat[::-1]
        y_hat_mask = y_hat_mask[::-1]
        y_mask = y_mask[::-1]
        # going backwards, the first non-zero alpha value should be the
        # first non-masked label.
        start_positions = T.cast(n_labels - y_mask.sum(0), 'int32')
    else:
        start_positions = T.zeros((batch_size,), dtype='int32')

    log_pred_y = _class_batch_to_labeling_batch(y, log_y_hat, y_hat_mask)
    log_pred_y = log_pred_y.dimshuffle(0, 2, 1)
    r2, r3 = _recurrence_relation(y, y_mask, blank_symbol)
    r2, r3 = T.log(r2), T.log(r3)

    def step(log_p_curr, y_hat_mask_t, log_p_prev):
        p1 = log_p_prev
        p2 = _log_dot_matrix(p1, r2)
        p3 = _log_dot_tensor(p1, r3)
        p12 = _log_add(p1, p2)
        p123 = _log_add(p3, p12)

        y_hat_mask_t = y_hat_mask_t[:, None]
        out = log_p_curr.T + p123 + T.log(y_mask.T)
        return _log_add(T.log(y_hat_mask_t) + out,
                        T.log(1 - y_hat_mask_t) + log_p_prev)

    log_probabilities, _ = theano.scan(
        step,
        sequences=[log_pred_y, y_hat_mask],
        outputs_info=[T.log(tensor.eye(n_labels)[start_positions])])

    return log_probabilities + T.log(y_hat_mask[:,:, None])


def _log_forward_backward(y, log_y_hat, y_mask, y_hat_mask, blank_symbol):
    log_probabs_forward = _log_path_probabs(y,
                                            log_y_hat,
                                            y_mask,
                                            y_hat_mask,
                                            blank_symbol)
    log_probabs_backward = _log_path_probabs(y,
                                             log_y_hat,
                                             y_mask,
                                             y_hat_mask,
                                             blank_symbol,
                                             reverse=True)
    return log_probabs_forward, log_probabs_backward[::-1][:,:, ::-1]


def _labeling_batch_to_class_batch(y, y_labeling, num_classes,
                                   y_hat_mask=None):
    # FIXME: y_hat_mask is currently not used
    batch_size = y.shape[1]
    N = y_labeling.shape[0]
    n_labels = y.shape[0]
    # sum over all repeated labels
    # from (T, B, L) to (T, C, B)
    out = T.zeros((num_classes, batch_size, N))
    y_labeling = y_labeling.dimshuffle((2, 1, 0))  # L, B, T
    y_ = y

    def scan_step(index, prev_res, y_labeling, y_):
        res_t = T.inc_subtensor(prev_res[y_[index, T.arange(batch_size)],
                                T.arange(batch_size)],
                                y_labeling[index, T.arange(batch_size)])
        return res_t

    result, updates = theano.scan(scan_step,
                                  sequences=[T.arange(n_labels)],
                                  non_sequences=[y_labeling, y_],
                                  outputs_info=[out])
    # result will be (C, B, T) so we make it (T, B, C)
    return result[-1].dimshuffle(2, 1, 0)
