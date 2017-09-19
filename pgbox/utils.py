from builtins import range
import pgbox.logging.logger as logger
import tensorflow as tf
import numpy as np
import scipy.signal
from pgbox.trpo.filters.filters import ZFilter

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

# KL divergence with itself, holding first argument fixed
def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)

# probability to take action x, given paramaterized guassian distribution
def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2.0*logstd)
    gp = -tf.square(x - mu)/(2.0*var) - .5*tf.log(tf.constant(2.0*np.pi)) - logstd
    return  tf.reduce_sum(gp, [1])

# KL divergence between two paramaterized guassian distributions
def gauss_KL(old_means, old_log_stds, new_means, new_log_stds):
    old_std = tf.exp(old_log_stds)
    new_std = tf.exp(new_log_stds)
    numerator = tf.square(old_means - new_means) + \
                tf.square(old_std) - tf.square(new_std)
    denominator = 2.0 * tf.square(new_std) + 1e-8
    return tf.reduce_sum(numerator / denominator + new_log_stds - old_log_stds)

class XavierUniformInitializer(object):
    def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
        if len(shape) == 2:
            n_inputs, n_outputs = shape
        else:
            receptive_field_size = np.prod(shape[:2])
            n_inputs = shape[-2] * receptive_field_size
            n_outputs = shape[-1] * receptive_field_size
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)


# Shannon entropy for a paramaterized guassian distributions
def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5*np.log(2.0*np.pi*np.e), tf.float32))
    return h

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def discount2(x, gamma):
    return x * (gamma ** np.arange(len(x)))

def discount_return(x, discount):
    return np.sum(x * (discount ** np.arange(len(x))))

def discount_backwards(x, discount):
    return np.cumsum(x * (discount ** np.arange(len(x))))

def cat_sample(prob_nk):
    assert prob_nk.ndim == 2
    # prob_nk: batchsize x actions
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(range(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def concat_tensor_list(tensor_list):
    return np.concatenate(tensor_list, axis=0)


# Immutable, lazily evaluated dict
class lazydict(object):
    def __init__(self, **kwargs):
        self._lazy_dict = kwargs
        self._dict = {}

    def __getitem__(self, key):
        if key not in self._dict:
            self._dict[key] = self._lazy_dict[key]()
        return self._dict[key]

    def __setitem__(self, i, y):
        self.set(i, y)

    def get(self, key, default=None):
        if key in self._lazy_dict:
            return self[key]
        return default

    def set(self, key, value):
        self._lazy_dict[key] = value


def extract(x, *keys):
    if isinstance(x, (dict, lazydict)):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError

def concat_tensor_dict_list(tensor_dict_list):
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        elif isinstance(example, list) or isinstance(example, tuple) or isinstance(example, np.ndarray):
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        else:
            v = np.array([x[k] for x in tensor_dict_list]).reshape(-1, 1)
        ret[k] = v
    return ret

def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def get_filters(args, observaton_space):
    if args.use_reward_filter:
        reward_filter = ZFilter((), demean=False, clip=10)
    else:
        reward_filter = lambda x : x

    if args.use_obs_filter:
        obs_filter = ZFilter(observaton_space.shape, clip=5)
    else:
        obs_filter = lambda x : x

    return obs_filter, reward_filter

def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list, clip=False):
    grads = tf.gradients(loss, var_list)
    if clip:
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
    for idx, (grad, param) in enumerate(zip(grads, var_list)):
        # If we hack around and stop the gradients, need to just zero them out
        if grad is None:
            grads[idx] = tf.zeros_like(param)
    return tf.concat(axis=0, values=[tf.reshape(grad, [numel(v)]) for (v, grad) in zipsame(var_list, grads)])

def conjugate_gradient(f_Ax, b, cg_iters=50, residual_tol=1e-10):
    # in numpy
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch2(f, x, fullstep, max_kl, max_backtracks=20, backtrack_ratio = .8, allow_backwards_steps=False):
    fval, kl = f(x)
    logger.log("fval/kl before %f/%f" % (fval, kl))

    for (_n_backtracks, stepfrac) in enumerate(backtrack_ratio ** np.arange(max_backtracks)):
        xnew = x - stepfrac * fullstep
        newfval, newkl = f(xnew)
        actual_improve = fval - newfval
        # expected_improve = expected_improve_rate * stepfrac
        # ratio = actual_improve / expected_improve
        # if ratio > accept_ratio and actual_improve > 0:
        logger.log(("a/kl %f/%f" % (actual_improve, newkl)))
        if not allow_backwards_steps:
            if newfval < fval and newkl <= max_kl:
                logger.log("backtrack iters: %d" % _n_backtracks)
                return xnew
        else:
            if newkl <= max_kl:
                logger.log("backtrack iters: %d" % _n_backtracks)
                return xnew
        logger.log("backtrack iters: %d" % _n_backtracks)
    return x

def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1, backtrack_ratio = .5):
    fval, kl = f(x)
    logger.log("fval before %f" % fval)

    for (_n_backtracks, stepfrac) in enumerate(backtrack_ratio ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval, kl = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        logger.log(("a/e/r %f/%f/%f" % (actual_improve, expected_improve, ratio)))
        if ratio > accept_ratio and actual_improve > 0:
            logger.log("fval after: %f"% (newfval))
            return xnew
    return x

class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(tf.assign(v,tf.reshape(self.theta[start:start + size],shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})

class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)

class GetPolicyWeights(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = [var for var in var_list if 'policy' in var.name]
    def __call__(self):
        return self.session.run(self.op)

class SetPolicyWeights(object):
    def __init__(self, session, var_list):
        self.session = session
        self.policy_vars = [var for var in var_list if 'policy' in var.name]
        self.placeholders = {}
        self.assigns = []
        for var in self.policy_vars:
            self.placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            self.assigns.append(tf.assign(var,self.placeholders[var.name]))
    def __call__(self, weights):
        feed_dict = {}
        count = 0
        for var in self.policy_vars:
            feed_dict[self.placeholders[var.name]] = weights[count]
            count += 1
        self.session.run(self.assigns, feed_dict)
