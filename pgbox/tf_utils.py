import tensorflow as tf
import numpy as np
from tensorflow.python.util import tf_contextlib

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def log10(x):
    numerator = tf.log(tf.clip_by_value(x, 1.0e-8, 99999))
    #TODO: Use a sensical value for upper bound    ^
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits_B):
    ent_B = (1.-tf.sigmoid(logits_B))*logits_B - logsigmoid(logits_B)
    return ent_B


def leaky_relu(x, alpha=.1):
    x = tf.maximum(x, alpha * x)
    return x

def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>0.0, x, alpha*tf.exp(x)-alpha)

def expand_multiple_dims(x, axes, name="expand_multiple_dims"):
  """
  :param tf.Tensor x:
  :param list[int]|tuple[int] axes: after completion, tf.shape(y)[axis] == 1 for axis in axes
  :param str name: scope name
  :return: y where we have a new broadcast axis for each axis in axes
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    for i in sorted(axes):
      x = tf.expand_dims(x, axis=i, name="expand_axis_%i" % i)
    return x


class GradInverter:
    def __init__(self, session, action_input, action_bounds, grad):
        """
        Reference: https://github.com/MOCR/
        https://raw.githubusercontent.com/stevenpjg/ddpg-aigym/master/tensorflow_grad_inverter.py
        """
        self.sess = session
        self.action_size = len(action_bounds[0])
        self.action_input = action_input
        self.pmax = tf.constant(action_bounds[0], dtype = tf.float32)
        self.pmin = tf.constant(action_bounds[1], dtype = tf.float32)
        self.prange = tf.constant([x - y for x, y in zip(action_bounds[0],action_bounds[1])], dtype = tf.float32)
        self.pdiff_max = tf.div(-self.action_input + self.pmax, self.prange)
        self.pdiff_min = tf.div(self.action_input - self.pmin, self.prange)
        self.zeros_act_grad_filter = tf.zeros([self.action_size])
        self.act_grad = grad#= tf.placeholder(tf.float32, [None, self.action_size])
        self.grad_inverter = tf.where(tf.greater(self.act_grad, self.zeros_act_grad_filter), tf.multiply(self.act_grad, self.pdiff_max), tf.multiply(self.act_grad, self.pdiff_min))

    def invert(self, grad, action):
        return self.sess.run(self.grad_inverter, feed_dict = {self.action_input: action})

def dimshuffle(x, axes, name="dimshuffle"):
  """
  Like Theanos dimshuffle.
  Combines tf.transpose, tf.expand_dims and tf.squeeze.

  :param tf.Tensor x:
  :param list[int|str]|tuple[int|str] axes:
  :param str name: scope name
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    assert all([i == "x" or isinstance(i, int) for i in axes])
    real_axes = [i for i in axes if isinstance(i, int)]
    bc_axes = [i for (i, j) in enumerate(axes) if j == "x"]
    if x.get_shape().ndims is None:
      x_shape = tf.shape(x)
      x = tf.reshape(x, [x_shape[i] for i in range(max(real_axes) + 1)])  # will have static ndims
    assert x.get_shape().ndims is not None

    # First squeeze missing axes.
    i = 0
    while i < x.get_shape().ndims:
      if i not in real_axes:
        x = tf.squeeze(x, axis=i)
        real_axes = [(j if (j < i) else (j - 1)) for j in real_axes]
      else:
        i += 1

    # Now permute.
    assert list(sorted(real_axes)) == list(range(x.get_shape().ndims))
    if real_axes != list(range(x.get_shape().ndims)):
      x = tf.transpose(x, real_axes)

    # Now add broadcast dimensions.
    if bc_axes:
      x = expand_multiple_dims(x, bc_axes)
    assert len(axes) == x.get_shape().ndims
    return x


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

def variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=tf.float32)
  return var


def variable_with_weight_decay(name, shape, stddev=1e-3, wd=0.01):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = variable_on_cpu(
      name,
      shape,
      initializer=tf.truncated_normal_initializer(
          stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
  return var

def optimized_moments(x, name=None):
    """Trying to create a more efficient moments that doesn't generate extra garbage in the graph
    """
    with tf.name_scope(name, "moments", [x, [1]]):
        # The dynamic range of fp16 is too limited to support the collection of
        # sufficient statistics. As a workaround we simply perform the operations
        # on 32-bit floats before converting the mean and variance back to fp16
        y = x#, tf.float32)
        N = len(x)
        # Compute true mean while keeping the dims for proper broadcasting.
        mean = tf.add_n(y, name="mean") / N

        mean_square = tf.square(mean)
        variance = tf.add_n([tf.square(z) for z in y], name="variance") / N - mean_square
        # sample variance, not unbiased variance
        # variance = tf.reduce_mean(tf.squared_difference(y, ), 1, keep_dims=True, name="variance")
        # if x.dtype == dtypes.float16:
        #   return (tf.cast(mean, tf.float16), tf.cast(variance, tf.float16))
        # else:
    return (mean, variance)



def bayesian_block(all_outs):
    """ returns mean and **variance** from mcdropout as a TF graph, can do uncertainty estimation in this way
    Problem with this current is that it gets:
        TypeError: Second-order gradient for while loops not supported.
    Due to the hvp approach (hessian second order deriviative)
        But looks like TF isn't going to support it anytime soon: https://github.com/tensorflow/tensorflow/issues/5985
    """
    # TODO: is dropout happening correctly, here?
    assert type(all_outs) is list

    # x = tf.stack(all_outs)
    # x = tf.concat([tf.reshape(out, (1,) + tuple(out.get_shape().as_list())) for out in all_outs], axis=0)
    # mean, var = tf.nn.moments(x, axes=[0])
    return optimized_moments(all_outs)

def gen_dropout_masks(mask_placeholders, keep_prob, seed=None):
    mask_dict = {}
    for mask in mask_placeholders:
        mask_dict[mask] = np.floor(keep_prob + np.random.uniform(size=(1, mask.get_shape()[-1])))
    return mask_dict

def dropout_with_mask(x, num_masks_so_far, keep_prob, mask=None, noise_shape=None, seed=None, name=None, reuse=False):  # pylint: disable=invalid-name
    with variable_scope("dropout", reuse=reuse) as name:
        # x = tf.convert_to_tensor(x, name="x")
        if mask is None:
            mask = tf.placeholder(tf.float32, x.get_shape(), name="mask%d" % num_masks_so_far)
        ret = tf.div(x, keep_prob) * mask
        ret.set_shape(x.get_shape())
    return ret, mask

def normalize_stats(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

# def variable_scope(name, reuse=None):
#     if reuse:
#         return reuse_name_scope(name)
#     else:
#         yield tf.variable_scope(name)

@tf_contextlib.contextmanager
def variable_scope(name, reuse=True, absolute=None):
  """
  TODO: this is literally the worst hack ever, tensorflow is the worst. should rewrite this in pytorch
  :param str|tf.VariableScope name: relative or absolute name scope (absolute if absolute=True or if tf.VariableScope).
    must not end with "/".
  :param bool absolute: if True it will be absolute
  We try to both set the variable scope and the name scope.
  """
  if isinstance(name, tf.VariableScope):
    name = name.name
    if absolute is not None:
      assert absolute is True
    absolute = True
  assert isinstance(name, str)
  if not absolute:
    assert name
    # First figure out the absolute name scope which we want to reuse/set.
    # The current name scope is more reliable because tf.variable_scope
    # will always also set the name scope.
    current_name_scope = get_current_name_scope()
    if current_name_scope:
      name = current_name_scope + "/" + name
  else:
    current_name_scope = None  # not needed
  assert name[-1:] != "/"
  abs_name = name + "/" if name else ""
  # tf.name_scope with a scope-name ending with "/" will interpret is as absolute name,
  # and use it as-is.
  # In all other cases, it would create a new name-scope with a new unique name,
  # which is not what we want.
  with tf.name_scope(abs_name):
    # tf.name_scope will not set the variable scope.
    # tf.variable_scope will also set the name scope, but the logic is broken
    # for absolute name scopes, thus we had to do the tf.name_scope manually above.
    # We create the dummy_var_scope to force it to reuse that name.
    # Note that the reuse-argument might be miss-leading in this context:
    # It means that tf.get_variable() will search for existing variables and errors otherwise.
    dummy_var_scope = tf.VariableScope(reuse=None, name=abs_name)
    with tf.variable_scope(dummy_var_scope) as scope:
      assert isinstance(scope, tf.VariableScope)
      # remove "/" from the end of the var-scope.
      # This is a work-around to fix up the variable scope behavior for nested variable scopes.
      # Warning: This might break at some future point.
      assert scope.name is scope._name
      assert scope.name[-1:] == "/" or scope.name == ""
      scope._name = scope._name[:-1]
      assert name == scope.name, "%r" % current_name_scope
      yield scope


def get_current_var_scope_name():
  """
  :return: current absolute variable scope name, via tf.variable_scope
  :rtype: str
  """
  v = tf.get_variable_scope()
  return v.name


def get_current_name_scope():
  """
  :return: current absolute name scope, via tf.name_scope
  :rtype: str
  http://stackoverflow.com/questions/40907769/how-to-get-current-tensorflow-name-scope
  Note that this is a private member and might break at some point.
  Note also that this does not need to be the same as get_current_var_scope_name().
  """
  return tf.get_default_graph()._name_stack or ""


def denormalize_stats(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean
