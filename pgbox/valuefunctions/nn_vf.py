import tensorflow as tf
import numpy as np
from pgbox.utils import *
from pgbox.tf_utils import *
from pgbox.trpo.filters.tf_rms import *


class MLPConstrainedValueFunction(object):
    """
    TODO: remove
    """
    coeffs = None
    def __init__(self, env, hidden_sizes=(64,64), activation=tf.nn.relu, scope="vf", max_iters = 25, mixfrac = .1 ):
        self.net = None
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.max_iters = max_iters
        self.timestep_limit = env.spec.timestep_limit
        self.observation_size = env.observation_space.shape[0]
        self.mixfrac = mixfrac
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size + 1], name="vf_obs")
        self.y = tf.placeholder(tf.float32, [None, 1], name="vf_y")
        self.old_y = tf.placeholder(tf.float32, [None, 1], name="old_vf_y")


        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        with tf.variable_scope(scope):
            net = self.obs
            for i, x in enumerate(hidden_sizes):
                net = tf.layers.dense(inputs=net, units=x, activation=activation, kernel_initializer=XavierUniformInitializer(), name="vf_h%d"%i)
            self.net = tf.layers.dense(inputs=net, units=1, activation=None, kernel_initializer=XavierUniformInitializer(), name="vf_outlayer")
            #self.net = tf.reshape(net, [-1])

        self.output = self.net

        # self.mse = tf.reduce_mean(tf.square(self.net - self.y))

        # var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        # self.l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in var_list if 'bias' not in v.name ]) * 1e-3
        # self.loss = self.mse + self.l2_loss
        # # self.train = tf.train.AdamOptimizer().minimize(l2)
        # self.train = tf.contrib.opt.ScipyOptimizerInterface(
        #     self.loss,
        #     var_list = var_list,
        #     method='L-BFGS-B',
        #     options={'maxiter': max_iters})

    def get_feed_vals(self, paths, sess):
        ob_no = np.concatenate([self.preproc(path["observations"]) for path in paths], axis=0)
        yrpredold = np.concatenate([self.predict(path, sess) for path in paths])#sess.run([self.net],  feed_dict={self.obs : ob_no })[0]
        yrpredold = yrpredold.reshape(-1, 1)
        vtarg_n1 = np.concatenate([path["returns"] for path in paths], axis=0)
        vtarg_n1 = vtarg_n1.reshape(-1,1)
        feed_dict = {self.old_y : yrpredold, self.obs : ob_no, self.y : vtarg_n1}
        return feed_dict

    def preproc(self, ob_no):
        return np.concatenate([ob_no, np.arange(len(ob_no)).reshape(-1,1) / float(self.timestep_limit)], axis=1)

    def predict(self, path, sess):
        ob_no = self.preproc(path["observations"])
        ret = sess.run(self.net, feed_dict = {self.obs : ob_no})[:,0]
        return ret


class MLPValueFunction(object):
    coeffs = None
    def __init__(self, env, hidden_sizes=(64,64), activation=tf.nn.relu, scope="vf", max_iters = 25, mixfrac = .1, use_rms_filter=False ):
        self.net = None
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.max_iters = max_iters
        self.timestep_limit = env.spec.timestep_limit
        self.observation_size = env.observation_space.shape[0]
        self.mixfrac = mixfrac
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size + 1], name="vf_obs")
        self.y = tf.placeholder(tf.float32, [None, 1], name="vf_y")


        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        obs_shape = list(env.observation_space.shape)
        obs_shape[-1] += 1

        with tf.variable_scope(scope):
            if use_rms_filter:
                with tf.variable_scope("obfilter"):
                    self.ob_rms = RunningMeanStd(shape=tuple(obs_shape))

                net = tf.clip_by_value((self.obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            else:
                net = self.obs
            for i, x in enumerate(hidden_sizes):
                net = tf.layers.dense(inputs=net, units=x, activation=activation, kernel_initializer=XavierUniformInitializer(), name="vf_h%d"%i)
            self.net = tf.layers.dense(inputs=net, units=1, activation=None, kernel_initializer=XavierUniformInitializer(), name="vf_outlayer")
            #self.net = tf.reshape(net, [-1])
        self.mse = tf.reduce_mean(tf.square(self.net - self.y))

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        self.l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in var_list if 'bias' not in v.name ]) * 1e-3
        self.loss = self.mse + self.l2_loss
        # self.train = tf.train.AdamOptimizer().minimize(l2)
        self.train = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            var_list = var_list,
            method='L-BFGS-B',
            options={'maxiter': max_iters})

    def preproc(self, ob_no):
        return np.concatenate([ob_no, np.arange(len(ob_no)).reshape(-1,1) / float(self.timestep_limit)], axis=1)

    def predict(self, path, sess):
        ob_no = self.preproc(path["observations"])
        ret = sess.run(self.net, feed_dict = {self.obs : ob_no})[:,0]
        return ret

    def fit(self, paths, sess):
        ob_no = np.concatenate([self.preproc(path["observations"]) for path in paths], axis=0)

        if hasattr(self, "ob_rms"): self.ob_rms.update(ob_no, sess)

        yrpredold = np.concatenate([self.predict(path, sess) for path in paths])#sess.run([self.net],  feed_dict={self.obs : ob_no })[0]
        vtarg_n1 = np.concatenate([path["returns"] for path in paths], axis=0)

        # from modular_rl
        ys = vtarg_n1*self.mixfrac + yrpredold*(1-self.mixfrac)
        ys = ys.reshape(-1, 1)

        loss, mse, l2, yrpredold = sess.run([self.loss, self.mse, self.l2_loss, self.net],  feed_dict={self.obs : ob_no, self.y : ys})
        logger.record_tabular("vf_loss_before", loss)
        logger.record_tabular("vf_mse_loss_before", mse)
        logger.record_tabular("vf_l2_loss_before", l2)

        self.train.minimize(sess, feed_dict = {self.obs : ob_no, self.y : ys})

        loss, mse, l2 = sess.run([self.loss, self.mse, self.l2_loss], feed_dict={self.obs : ob_no, self.y : ys})
        logger.record_tabular("vf_loss_after", loss)
        logger.record_tabular("vf_mse_after", mse)
        logger.record_tabular("vf_l2_after", l2)
        return loss


class MLPActionValueFunction(object):
    coeffs = None
    def __init__(self, env, hidden_sizes=(64,64), activation=tf.nn.relu, scope="vf", update_method="adam", max_iters = 10, mixfrac = 1.0, use_rms_filter=False, output_activation=None, shared_rms_filter=None ):
        self.net = None
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.max_iters = max_iters
        self.timestep_limit = env.spec.timestep_limit
        self.observation_size = env.observation_space.shape[0]
        self.env = env
        self.action_size = env.action_space.shape[0]
        self.update_method = update_method
        self.mixfrac = mixfrac
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size], name="vf_obs")
        self.action = tf.placeholder(tf.float32, [None, self.action_size], name="vf_obs")
        self.y = tf.placeholder(tf.float32, [None, 1], name="vf_y")


        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        obs_shape = list(env.observation_space.shape)
        obs_shape[-1] += 1
        self.scope = scope
        self.use_rms_filter = use_rms_filter
        self.shared_rms_filter = shared_rms_filter
        self.ob_rms = None
        self.hidden_sizes = hidden_sizes
        self.net = self._make_net(self.obs, self.action, hidden_sizes, activation, output_activation, reuse=False, scope=scope)
            #self.net = tf.reshape(net, [-1])
        self.activation = activation
        self.output_activation = output_activation

    def apply_adam_with_gradient_clipping(self, var_list, lr=1e-3, clip_param=1.5, iters=1):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        grads = self.optimizer.compute_gradients(self.loss, var_list=var_list)
        grads = [(tf.clip_by_value(grad, -clip_param, clip_param), var) for grad, var in grads if grad is not None]
        # grads_and_vars = list(zip(grads, policy_varlist))
        self.optim = self.optimizer.apply_gradients(grads)
        return lambda sess,feed :[sess.run([self.optim], feed_dict = feed) for i in range(2)]



    def init_optimizer(self, target, output):
        self.td_error = tf.square(output - target)
        self.mse = tf.reduce_mean(self.td_error)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

        self.l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in var_list if 'bias' not in v.name ]) * 1e-3
        self.loss = self.mse + self.l2_loss

        # self.train = tf.train.AdamOptimizer().minimize(l2)
        if self.update_method == "adam":
            # self.optim = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss, var_list=var_list)
            # TODO: replace mixfrac with trust region
            self.train = self.apply_adam_with_gradient_clipping(var_list)
        else:
            self.train = lambda sess, feed : tf.contrib.opt.ScipyOptimizerInterface(
                self.loss,
                var_list = var_list,
                method='L-BFGS-B',
                options={'maxiter': self.max_iters}).minimize(sess, feed_dict = feed)

    def create_dupe(self, scope):
        return MLPActionValueFunction(self.env,
                                      self.hidden_sizes,
                                      activation=self.activation,
                                      scope=scope,
                                      max_iters=self.max_iters,
                                      mixfrac = self.mixfrac,
                                      use_rms_filter=self.use_rms_filter,
                                      output_activation=self.output_activation,
                                      shared_rms_filter=self.ob_rms,
                                      update_method=self.update_method)


    def filter_regularizable(self, var_list):
        var_list = [var for var in var_list if ('kernel' in var.name and not ('bias' in var.name) and not ('Adam' in var.name))]
        return var_list

    def filter_trainable(self, var_list):
        var_list = [var for var in var_list if (('kernel' in var.name or 'bias' in var.name or 'bn' in var.name or 'ln' in var.name) and not ('Adam' in var.name))]
        return var_list

    def get_params(self, trainable=False, regularizable=False):

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

        var_list = [var for var in var_list if not ('Adam' in var.name)]

        if trainable:
            var_list = self.filter_trainable(var_list)
        if regularizable:
            var_list = self.filter_regularizable(var_list)

        return var_list

    def get_param_values(self, sess, trainable=False, regularizable=False):
        return sess.run(self.get_params(trainable=trainable, regularizable=regularizable))

    def _make_net(self, observation, action, hidden_sizes, activation, output_activation, reuse, scope):
        with variable_scope(scope, reuse=reuse):
            if self.use_rms_filter:
                if not hasattr(self, "ob_rms") and self.shared_rms_filter is None:
                    with variable_scope("obfilter"):
                        self.ob_rms = RunningMeanStd(shape=tuple(obs_shape))
                elif not (self.shared_rms_filter is None):
                    self.ob_rms = self.shared_rms_filter
                net = tf.clip_by_value((observation - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            else:
                net = observation
            net = tf.concat([net, action], axis=-1)
            for i, x in enumerate(hidden_sizes):
                net = tf.layers.dense(inputs=net, units=x, activation=activation, kernel_initializer=XavierUniformInitializer(), name="vf_h%d"%i)
            net = tf.layers.dense(inputs=net, units=1, activation=output_activation, kernel_initializer=XavierUniformInitializer(), name="vf_outlayer")
        return net

    def predict(self, observation, actions, sess, extra_feed = {}):
        feed = {self.obs : observation, self.action: actions}
        feed.update(extra_feed)
        ret = sess.run(self.net, feed_dict = feed)[:,0]
        return ret

    def get_qval(self, observation, actions, sess):
        return self.predict(observation, actions, sess)

    def get_symbolic_with_action(self, new_action_input):
        """
        replaces the action input of the q function with a simple
        """
        net = self._make_net(self.obs, new_action_input, self.hidden_sizes, self.activation, self.output_activation, reuse=True, scope=self.scope)
        return net

    def fit(self, ob_no, actions, ys, sess, extra_feed = {}):
        if hasattr(self, "ob_rms") and not (self.ob_rms is None): self.ob_rms.update(ob_no, sess)

        yrpredold = self.predict(ob_no, actions, sess, extra_feed=extra_feed).reshape(-1, 1)#sess.run([self.net],  feed_dict={self.obs : ob_no })[0]
        vtarg_n1 = ys

        # from modular_rl
        feed = {self.obs : ob_no, self.action: actions, self.y : ys}
        feed.update(extra_feed)
        ys = vtarg_n1*self.mixfrac + yrpredold*(1.0-self.mixfrac)
        ys = ys.reshape(-1, 1)

        loss, mse, l2, yrpredold = sess.run([self.loss, self.mse, self.l2_loss, self.net],  feed_dict=feed)
        # logger.record_tabular("vf_loss_before", loss)
        # logger.record_tabular("vf_mse_loss_before", mse)
        # logger.record_tabular("vf_l2_loss_before", l2)

        self.train(sess, feed)

        td_error, loss, mse, l2 = sess.run([self.td_error, self.loss, self.mse, self.l2_loss], feed_dict=feed)
        # logger.record_tabular("vf_loss_after", loss)
        # logger.record_tabular("vf_mse_after", mse)
        # logger.record_tabular("vf_l2_after", l2)
        return td_error, loss, l2, yrpredold, []
