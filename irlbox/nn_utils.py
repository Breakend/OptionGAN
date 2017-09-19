import tensorflow as tf
import numpy as np
from pgbox.tf_utils import *

class Standardizer(object):
    def __init__(self, dim, eps=1e-6, init_count=0, init_mean=0., init_meansq=1.):
        '''
        Args:
            dim: dimension of the space of points to be standardized
            eps: small constant to add to denominators to prevent division by 0
            init_count, init_mean, init_meansq: initial values for accumulators
        Note:
            if init_count is 0, then init_mean and init_meansq have no effect beyond
            the first call to update(), which will ignore their values and
            replace them with values from a new batch of data.
        '''
        self.sess = tf.Session()
        self._eps = tf.constant(eps, dtype=tf.float32)
        self._dim = tf.constant(dim, dtype=tf.float32)
        self._count = tf.Variable(init_count, name="count", trainable=False)
        # import pdb; pdb.set_trace()
        self._mean_1_D = tf.Variable(np.full((1, dim), init_mean), name="mean_1_D", trainable=False, dtype=tf.float32)
        self._meansq_1_D = tf.Variable(np.full((1, dim), init_meansq), name="meansq_1_D", trainable=False, dtype=tf.float32)
        self._stdev_1_D = tf.sqrt(tf.nn.relu(self._meansq_1_D - tf.square(self._mean_1_D)))
        self.points_N_D = tf.placeholder(tf.float32, [None, dim])
        self.batch_num = tf.placeholder(tf.int32, [])
        # Relu ensures inside is nonnegative. maybe the better choice would have been to
        # add self._eps inside the square root, but I'm keeping things this way to preserve
        # backwards compatibility with existing saved models.
        # self.get_mean = lambda () : self.sess.run([self._mean_1_D])
        # self.get_stdev = lambda () : self.sess.run([self._stdev_1_D])[0,:]
        a = tf.cast(tf.divide(self._count, (self._count + self.batch_num)), tf.float32)
        assign_ops = []
        assign_ops.append(self._mean_1_D.assign(a*self._mean_1_D + (1.-a)*tf.reduce_mean(self.points_N_D, axis=0)))
        assign_ops.append(self._meansq_1_D.assign(a*self._meansq_1_D + (1.-a)* tf.reduce_mean(tf.square(self.points_N_D), axis=0)))
        assign_ops.append(self._count.assign(self._count + self.batch_num))
        self.assign_ops = assign_ops
        initialize_uninitialized(self.sess)

    @property
    def varscope(self):
        return self.__varscope

    def update(self, points):
        num = points.shape[0]
        return self.sess.run(self.assign_ops, feed_dict = {self.points_N_D : points, self.batch_num : num})

    def standardize_expr(self, x_B_D):
        # import pdb; pdb.set_trace()
        return (x_B_D - self._mean_1_D) / (self._stdev_1_D + self._eps)

    def unstandardize_expr(self, y_B_D):
        return y_B_D*(self._stdev_1_D + self._eps) + self._mean_1_D

    def standardize(self, x_B_D):
        assert x_B_D.ndim == 2
        return (x_B_D - self.get_mean()) / (self.get_stdev() + self._eps)

    def unstandardize(self, y_B_D):
        assert y_B_D.ndim == 2
        return y_B_D*(self.get_stdev() + self._eps) + self.get_mean()
