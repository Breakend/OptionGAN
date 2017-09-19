import numpy as np
import tensorflow as tf
from pgbox.utils import *
from pgbox.distributions.gaussian import Gaussian
from pgbox.trpo.filters.tf_rms import *
from pgbox.tf_utils import variable_scope

class GaussianMLPPolicy(object):

    def __init__(self, env, hidden_sizes=(400,300), scope="policy", activation=tf.nn.relu, init_std=1.0, min_std=1e-6, batch_norm=False, use_rms_filter=False, network_std =False):

        # Hidden layer sizes of the mlp
        self.hidden_sizes = hidden_sizes
        # environment to gather info
        self.env = env
        self.activation = activation
        # TF variable scope
        self.scope = scope
        # number of duplicates made of this policy so far (for debugging parallel stuff)
        self.num_dupes = 0
        init_std_param = np.log(init_std)
        init_std_constant = tf.constant_initializer(init_std_param)
        self.min_std_param = np.log(min_std)

        # Observation sizes, assumes flattened for now, not compatible with Atari
        # TODO: make compatible with atari
        self.observation_size = env.observation_space.shape[0]
        self.action_size = np.prod(env.action_space.shape)
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size], name="policy_obs")
        self.action = tf.placeholder(tf.float32, [None, self.action_size], name="policy_action")

        self.distribution = Gaussian(env.action_space.shape[0])

        self.advantage = tf.placeholder(tf.float32, [None], name="policy_advantage")
        self.oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size], name="oldaction_dist_mu")
        self.oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size], name="oldaction_dist_logstd")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.batch_norm = batch_norm
        self.var_list = None

        with variable_scope(scope):

            if use_rms_filter:
                with variable_scope("obfilter"):
                    self.ob_rms = RunningMeanStd(shape=env.observation_space.shape)

                net = tf.clip_by_value((self.obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            else:
                net = self.obs

            for i, x in enumerate(hidden_sizes):
                if self.batch_norm:
                    net = tf.layers.batch_normalization(net, training=self.phase_train, name="policy_bn%d"%i)
                net = tf.layers.dense(inputs=net, units=x, activation=activation, kernel_initializer=XavierUniformInitializer(), name="policy_h%d"%i)
            net = tf.layers.dense(inputs=net, units=self.action_size, activation=None, kernel_initializer=XavierUniformInitializer(), name="policy_outlayer")
            self.action_dist_mu = net
            if network_std:
                self.action_dist_logstd = action_dist_logstd_param = tf.layers.dense(inputs=net, units=self.action_size, activation=None, kernel_initializer=XavierUniformInitializer(), name="policy_logstd")
            else:
                action_dist_logstd_param = tf.get_variable(name="policy_logstd_kernel", shape=(1, self.action_size), initializer=init_std_constant, regularizer=None, dtype=tf.float32)
                self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

        # means for each action

        # log standard deviations for each actions
        # set a min std for numerical stability
        self.action_dist_logstd = tf.maximum(self.action_dist_logstd , self.min_std_param)

        self.batch_size_float = tf.cast(tf.shape(self.obs)[0], tf.float32)

        # what are the probabilities of taking self.action, given new and old distributions
        self.log_p_n = self.distribution.log_likelihood(self.action_dist_mu, self.action_dist_logstd, self.action)
        self.log_oldp_n = self.distribution.log_likelihood(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action)

        self.likelihood_ratio = tf.exp(self.log_p_n - self.log_oldp_n)
        self.ent = self.distribution.entropy(self.action_dist_mu, self.action_dist_logstd)
        self.kl = self.distribution.kl(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action_dist_mu, self.action_dist_logstd)
        self.kl_firstfixed = self.distribution.kl(tf.stop_gradient(self.action_dist_mu), tf.stop_gradient(self.action_dist_logstd), self.action_dist_mu, self.action_dist_logstd)
        self.set_external_values = lambda x,y : None
        self.placeholders = {}
        self.assigns = []
        for var in self.get_params():
            self.placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            self.assigns.append(tf.assign(var,self.placeholders[var.name]))

    def filter_trainable(self, var_list):
        var_list = [var for var in var_list if (('kernel' in var.name or 'bias' in var.name) and not ('Adam' in var.name))]
        return var_list

    def get_params(self, trainable=False, regularizable=False, optimizer_params=True):

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

        var_list = self.var_list
        if trainable:
            var_list = self.filter_trainable(var_list)
        if not optimizer_params:
            var_list = [var for var in var_list if (not ('Adam' in var.name))]
        return var_list

    def get_param_values(self, sess, trainable=False, regularizable=False, optimizer_params=True):
        return sess.run(self.get_params(trainable=trainable, regularizable=regularizable, optimizer_params=optimizer_params))

    def set_param_values(self, sess, weights, trainable=False):
        var_list = self.get_params(trainable=trainable)
        assert len(weights) == len(var_list)
        feed_dict = {}
        count = 0
        for var in var_list:
            feed_dict[self.placeholders[var.name]] = weights[count]
            count += 1
        sess.run(self.assigns, feed_dict)

    def get_extra_inputs(self, sess, observations, infos):
        return {self.phase_train : True}

    def get_action_sym(self):
        return self.action_dist_mu + tf.exp(self.action_dist_logstd) * tf.random_normal(tf.shape(self.action_dist_mu))

    def act(self, obs, sess):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs: obs, self.phase_train : False})
        # samples the guassian distribution
        rnd = np.random.normal(size=action_dist_mu.shape)
        act = action_dist_mu + np.exp(action_dist_logstd) * rnd
        info = {"action_dist_mu" : action_dist_mu, "action_dist_logstd" : action_dist_logstd}
        return act.ravel(), info

    def get_actions(self, obs, sess):
        # obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs: obs, self.phase_train : False})
        # samples the guassian distribution
        rnd = np.random.normal(size=action_dist_mu.shape)
        act = action_dist_mu + np.exp(action_dist_logstd) * rnd
        info = {"action_dist_mu" : action_dist_mu, "action_dist_logstd" : action_dist_logstd}
        return act, info

    def create_dupe(self):
        self.num_dupes += 1
        return GaussianMLPPolicy(self.env, self.hidden_sizes, activation=self.activation, scope="pol_dupe_%d" % self.num_dupes)
