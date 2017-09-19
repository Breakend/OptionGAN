import numpy as np
import tensorflow as tf

TINY = 1e-8


def from_onehot(x_var):
    ret = np.zeros((len(x_var),), 'int32')
    nonzero_n, nonzero_a = np.nonzero(x_var)
    ret[nonzero_n] = nonzero_a
    return ret


class Categorical(object):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_prob_var, new_prob_var):
        """
        Compute the symbolic KL divergence of two categorical distributions
        """
        ndims = old_prob_var.get_shape().ndims
        # Assume layout is N * A
        return tf.reduce_sum(
            old_prob_var * (tf.log(old_prob_var + TINY) - tf.log(new_prob_var + TINY)),
            axis=ndims - 1
        )

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two categorical distributions
        """
        return np.sum(
            old_prob * (np.log(old_prob + TINY) - np.log(new_prob + TINY)),
            axis=-1
        )

    def likelihood_ratio_sym(self, x_var, old_prob_var, new_prob_var):
        ndims = old_prob_var.get_shape().ndims
        x_var = tf.cast(x_var, tf.float32)
        # Assume layout is N * A
        return (tf.reduce_sum(new_prob_var * x_var, ndims - 1) + TINY) / \
               (tf.reduce_sum(old_prob_var * x_var, ndims - 1) + TINY)

    def entropy_sym(self, dist_info_vars):
        return -tf.reduce_sum(probs * tf.log(probs + TINY), axis=1)

    def cross_entropy_sym(self, old_prob_var, new_prob_var):
        ndims = old_prob_var.get_shape().ndims
        # Assume layout is N * A
        return tf.reduce_sum(
            old_prob_var * (- tf.log(new_prob_var + TINY)),
            axis=ndims - 1
        )

    def entropy(self, probs):
        return -np.sum(probs * np.log(probs + TINY), axis=1)

    def log_likelihood_sym(self, x_var, probs):
        ndims = probs.get_shape().ndims
        return tf.log(tf.reduce_sum(probs * tf.cast(x_var, tf.float32), ndims - 1) + TINY)

    def log_likelihood(self, xs, probs):
        # Assume layout is N * A
        return np.log(np.sum(probs * xs, axis=-1) + TINY)

    @property
    def dist_info_specs(self):
        return [("prob", (self.dim,))]
