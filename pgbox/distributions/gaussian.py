import tensorflow as tf
import numpy as np

class Gaussian(object):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl(self, old_means, old_log_stds, new_means, new_log_stds):
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)

        var1 = tf.exp(tf.constant(2.0)*old_log_stds)
        var2 = tf.exp(tf.constant(2.0)*new_log_stds)

        kl = new_log_stds - old_log_stds + (var1 + tf.square(old_means - new_means))/(tf.constant(2.0)*var2) - tf.constant(0.5)
        return kl#, tf.reduce_mean(kl)

    def likelihood_ratio(self, old_means, old_stds, new_means, new_stds, x_var):
        logli_new = self.log_likelihood(new_means, new_stds, x_var)
        logli_old = self.log_likelihood(old_means, old_stds, x_var)
        return tf.exp(logli_new - logli_old)

    def log_likelihood(self, means, log_stds, x_var):
        var = tf.exp(tf.constant(2.0)*log_stds)
        gp = -tf.square(x_var - means)/(tf.constant(2.0)*var) - tf.constant(.5)*tf.log(tf.constant(2.0*np.pi)) - log_stds
        summed_gp = tf.reduce_sum(gp, [1])
        return summed_gp

    def sample(self, means, log_stds):
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def entropy(self, mu, log_stds):
        h = tf.reduce_sum(log_stds + tf.constant(0.5 * np.log(2.0 * np.pi * np.e), tf.float32), axis=-1)
        return h
