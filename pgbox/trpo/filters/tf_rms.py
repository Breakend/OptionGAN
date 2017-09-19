from mpi4py import MPI
import tensorflow as tf, numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):

        self._sum = tf.get_variable(
            dtype=tf.float32,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float32,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float32,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt( tf.maximum( tf.to_float(self._sumsq / self._count) - tf.square(self.mean) , 1e-2 ))

        self.newsum = tf.placeholder(shape=self.shape, dtype=tf.float32, name='sum')
        self.newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float32, name='var')
        self.newcount = tf.placeholder(shape=[], dtype=tf.float32, name='count')
        self.ops = [tf.assign_add(self._sum, self.newsum),
                 tf.assign_add(self._sumsq, self.newsumsq),
                 tf.assign_add(self._count, self.newcount)]


    def update(self, x, session):
        x = x.astype('float32')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float32')
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float32')])
        MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        feed_dict = {self.newsum : totalvec[0:n].reshape(self.shape),
                     self.newsumsq : totalvec[n:2*n].reshape(self.shape),
                     self.newcount : totalvec[2*n]
                     }
        session.run(self.ops, feed_dict)
