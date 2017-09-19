import numpy as np
import tensorflow as tf
from pgbox.utils import *
from pgbox.distributions.gaussian import Gaussian
from pgbox.trpo.filters.tf_rms import *

class GatedGaussianMLPPolicy(object):

    def __init__(self, env, hidden_sizes=(400,300), scope="policy", activation=tf.nn.relu, init_std=1.0, min_std=1e-6, num_options=2, gate_hidden_sizes=(64,64), use_rms_filter=False, use_shared_layer=True):
        #TODO: update create_dupe_script, or really just create a class above it that shares everything

        # Hidden layer sizes of the mlp
        self.hidden_sizes = hidden_sizes
        self.gate_hidden_sizes=gate_hidden_sizes
        # environment to gather info
        self.env = env
        self.num_options = num_options
        self.min_std = min_std
        self.use_shared_layer= use_shared_layer
        self.init_std=init_std
        # we are given a policy over options by an outside source that shouldn't be updated
        self.activation = activation
        self.use_rms_filter = use_rms_filter
        # TF variable scope
        self.var_list = None
        self.scope = scope
        # number of duplicates made of this policy so far (for debugging parallel stuff)
        self.num_dupes = 0

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
        self._make_net(env, hidden_sizes, scope, activation, init_std, min_std, num_options, gate_hidden_sizes, use_rms_filter, use_shared_layer)

    def _make_net(self,
                  env,
                  hidden_sizes,
                  scope,
                  activation,
                  init_std,
                  min_std,
                  num_options,
                  gate_hidden_sizes,
                  use_rms_filter,
                  use_shared_layer,
                  reuse=False,
                  extra_options=0,
                  stop_gate_gradient=True,
                  stop_old_option_gradients=False):

        means = []
        stds = []
        self.old_option_stds = []
        self.old_option_means = []
        for i in range(num_options):
            oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size], name="oldaction_dist_mu_o%d"%i)
            oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size], name="oldaction_dist_logstd_o%d"%i)
            self.old_option_stds.append(oldaction_dist_logstd)
            self.old_option_means.append(oldaction_dist_mu)
        init_std_param = np.log(init_std)
        init_std_constant = tf.constant_initializer(init_std_param)
        with tf.variable_scope(scope):
            if use_rms_filter:
                with tf.variable_scope("obfilter", reuse=reuse):
                    self.ob_rms = RunningMeanStd(shape=env.observation_space.shape)

                net_input = tf.clip_by_value((self.obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            else:
                net_input = self.obs

            # TODO: this is a hack right now that requires the gate to be the same, so we need to fix this to be properly passed a network and make a local copy.
            with tf.variable_scope("gate", reuse=reuse):
                gating_network = net_input
                for i, x in enumerate(gate_hidden_sizes):
                    gating_network = tf.layers.dense(inputs=gating_network, units=x, activation=activation, kernel_initializer= XavierUniformInitializer(), name="gating_hidden%d"%i, reuse=reuse)
            with tf.variable_scope("gate", reuse=None):
                gating_network = tf.layers.dense(inputs=gating_network, units=num_options, activation=tf.nn.softmax, kernel_initializer=XavierUniformInitializer(), name="gating_outlayer_%d"%extra_options, reuse=False)
                if stop_gate_gradient:
                    self.gate = tf.stop_gradient(gating_network)
                else:
                    self.gate = gating_network
                print(self.gate)

            mean_net = tf.constant(0.0)
            std_net = tf.constant(0.0)
            means, stds = [],[]
            if use_shared_layer:
                shared_net = net_input
                for i, x in enumerate(hidden_sizes):
                    shared_net = tf.layers.dense(inputs=shared_net, units=x, activation=activation, kernel_initializer=XavierUniformInitializer(), name="policy_h%d"%i,reuse=reuse)
                for o in range(num_options):
                    with tf.variable_scope("option%d" % o):
                        net = tf.layers.dense(inputs=shared_net, units=self.action_size, activation=None, kernel_initializer=XavierUniformInitializer(), name="policy_outlayer", reuse=(reuse and o < num_options - extra_options))
                    with tf.variable_scope("option%d" % o, reuse=(reuse and o < num_options - extra_options)):
                        action_dist_logstd_param = tf.get_variable(name="policy_logstd_kernel", shape=(1, self.action_size), initializer=init_std_constant, regularizer=None, dtype=tf.float32)
                        action_dist_logstd_param = tf.maximum(action_dist_logstd_param, self.min_std_param)
                        action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(net)[0], 1)))
                        if stop_old_option_gradients and o < num_options - extra_options:
                            print("stopping gradients on option %d" % o)
                            net = tf.stop_gradient(net)
                            action_dist_logstd = tf.stop_gradient(action_dist_logstd)
                        mean_net += net * tf.reshape(self.gate[:,o], (-1,1))
                        std_net += action_dist_logstd * tf.reshape(self.gate[:,o], (-1,1))
                        means.append(net)
                        stds.append(action_dist_logstd)
            else:
                for o in range(num_options):
                    with tf.variable_scope("option%d" % o):
                        net = net_input
                        for i, x in enumerate(hidden_sizes):
                            net = tf.layers.dense(inputs=net, units=x, activation=activation, kernel_initializer=XavierUniformInitializer(), name="policy_h%d"%i, reuse=(reuse and o < num_options - extra_options))
                    with tf.variable_scope("option%d" % o, reuse=(reuse and o < num_options - extra_options)):
                        net = tf.layers.dense(inputs=net, units=self.action_size, activation=None, kernel_initializer=XavierUniformInitializer(), name="policy_outlayer", reuse=(reuse and o < num_options - extra_options))
                        action_dist_logstd_param = tf.get_variable(name="policy_logstd_kernel", shape=(1, self.action_size), initializer=init_std_constant, regularizer=None, dtype=tf.float32)
                        action_dist_logstd_param = tf.maximum(action_dist_logstd_param, self.min_std_param)
                        action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(net)[0], 1)))
                        if stop_old_option_gradients and o < num_options - extra_options:
                            net = tf.stop_gradient(net)
                            action_dist_logstd = tf.stop_gradient(action_dist_logstd)
                        mean_net += net * tf.reshape(self.gate[:,o], (-1,1))
                        std_net += action_dist_logstd * tf.reshape(self.gate[:,o], (-1,1))
                        means.append(net)
                        stds.append(action_dist_logstd)

            self.termination_importance_values = tf.reduce_sum(self.gate, axis=0)

        # means for each action
        self.action_dist_mu = mean_net
        self.action_dist_logstd = std_net
        self.option_stds = stds
        self.option_means = means

        # log standard deviations for each actions
        # set a min std for numerical stability
        # tile for simplicity, only use one std predictor

        batch_size_float = tf.cast(tf.shape(self.obs)[0], tf.float32)

        # what are the probabilities of taking self.action, given new and old distributions
        self.log_p_n = self.distribution.log_likelihood(self.action_dist_mu, self.action_dist_logstd, self.action)
        self.log_oldp_n = self.distribution.log_likelihood(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action)

        self.likelihood_ratio = tf.exp(self.log_p_n - self.log_oldp_n)
        self.ent = self.distribution.entropy(self.action_dist_mu, self.action_dist_logstd)
        self.kl = self.distribution.kl(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action_dist_mu, self.action_dist_logstd)
        self.kl_firstfixed = self.distribution.kl(tf.stop_gradient(self.action_dist_mu), tf.stop_gradient(self.action_dist_logstd), self.action_dist_mu, self.action_dist_logstd)
        self.set_external_values = self.set_external_values_function()
        self.option_log_p_ns = []
        self.option_log_oldp_n = []
        self.option_kl = []
        self.option_kl_firstfixed = []
        self.option_likelihood_ratio = []
        for old_mean, old_std, mean, std in zip(self.old_option_means, self.old_option_stds, self.option_means, self.option_stds):
            option_log_p_n = self.distribution.log_likelihood(mean, std, self.action)
            option_log_oldp_n = self.distribution.log_likelihood(old_mean, old_std, self.action)
            self.option_log_p_ns.append(option_log_p_n)
            self.option_log_oldp_n.append(option_log_oldp_n)

            self.option_likelihood_ratio.append(tf.exp(option_log_p_n - option_log_oldp_n))
            self.option_kl.append(self.distribution.kl(old_mean, old_std, mean, std))
            self.option_kl_firstfixed.append(self.distribution.kl(tf.stop_gradient(mean), tf.stop_gradient(std), mean, std))

        self.placeholders = {}
        self.assigns = []
        for var in self.get_params():
            self.placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            self.assigns.append(tf.assign(var,self.placeholders[var.name]))

    def rebuild_net(self, extra_options=0, stop_old_option_gradients=False, stop_gate_gradient=True):
        self.num_options += extra_options
        self._make_net(self.env,
                       self.hidden_sizes,
                       self.scope,
                       self.activation,
                       self.init_std,
                       self.min_std,
                       self.num_options,
                       self.gate_hidden_sizes,
                       self.use_rms_filter,
                       self.use_shared_layer,
                       extra_options=extra_options,
                       reuse=True,
                       stop_old_option_gradients = stop_old_option_gradients,
                       stop_gate_gradient=stop_gate_gradient)

    def get_param_values(self, sess, trainable=False, regularizable=False, optimizer_params=True):
        return sess.run(self.get_params(trainable=trainable, regularizable=regularizable, optimizer_params=optimizer_params))

    def set_param_values(self, sess, weights):
        var_list = self.get_params()
        print("VarList %d and Weights %d" % (len(var_list), len(weights)))
        assert len(weights) == len(var_list)
        feed_dict = {}
        count = 0
        for var in var_list:
            feed_dict[self.placeholders[var.name]] = weights[count]
            count += 1
        sess.run(self.assigns, feed_dict)

    def filter_trainable(self, var_list):
        var_list = [var for var in var_list if (('kernel' in var.name or 'bias' in var.name) and not ('Adam' in var.name))]
        return var_list

    def get_params(self, trainable=False, regularizable=False, optimizer_params=True):
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "/")

        var_list = self.var_list
        if trainable:
            var_list = self.filter_trainable(var_list)
        if not optimizer_params:
            var_list = [var for var in var_list if (not ('Adam' in var.name))]

        return var_list

    # def set_gate_values():
    # TODO: need to make a copy of the gating values such that they are updated in the policy as well.
    def set_external_values_function(self):
        var_list = self.get_params()#tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "/")
        var_list = [var for var in var_list if (('gate' in var.name or 'filter' in var.name) and 'policy' in var.name)]
        placeholders = {}
        assigns = []
        for var in var_list:
            placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            assigns.append(tf.assign(var, placeholders[var.name]))
        def set_function(session, weights):
            feed_dict = {}
            count = 0
            placeholders_needed = list(placeholders.keys())
            for (varname, weight) in weights:
                # print("Setting %s to " % varname, weight)
                last_var_name = "/".join(varname.split("/")[1:])
                last_var_name = self.scope + "/" + last_var_name
                feed_dict[placeholders[last_var_name]] = weight
                placeholders_needed.remove(last_var_name)
                count += 1
            if len(placeholders_needed) > 0:
                raise Exception("Missing placeholders: %s" % ' '.join(placeholders_needed))
            session.run(assigns, feed_dict)
        return set_function

    def get_extra_inputs(self, sess, observations, infos):
        feed_dict = {}
        i = 0
        for mean, std in zip(self.old_option_means, self.old_option_stds):
            feed_dict.update({mean: infos["action_dist_mu_o%d"%i], std: infos["action_dist_logstd_o%d"%i] })
            i += 1
        return feed_dict

    def act(self, obs, sess):
        obs = np.expand_dims(obs, 0)

        action_dist_mu, action_dist_logstd, gate_dist = sess.run([self.action_dist_mu, self.action_dist_logstd, self.gate], feed_dict={self.obs: obs})
        # samples the guassian distribution
        rnd = np.random.normal(size=action_dist_mu.shape)
        act = action_dist_mu + np.exp(action_dist_logstd) * rnd
        info = {"action_dist_mu" : action_dist_mu, "action_dist_logstd" : action_dist_logstd, "gate_dist" :  gate_dist}
        i = 0
        for mean, std in zip(self.option_means, self.option_stds):
            action_dist_mu, action_dist_logstd = sess.run([mean, std], feed_dict={self.obs: obs})
            info.update({"action_dist_mu_o%d"%i : action_dist_mu, "action_dist_logstd_o%d"%i : action_dist_logstd })
            i += 1
        return act.ravel(), info

    def get_acts(self, obs, sess):
        action_dist_mu, action_dist_logstd, gate_dist = sess.run([self.action_dist_mu, self.action_dist_logstd, self.gate], feed_dict={self.obs: obs})
        # samples the guassian distribution
        rnd = np.random.normal(size=action_dist_mu.shape)
        act = action_dist_mu + np.exp(action_dist_logstd) * rnd
        info = {"action_dist_mu" : action_dist_mu, "action_dist_logstd" : action_dist_logstd, "gate_dist" :  gate_dist}
        i = 0
        for mean, std in zip(self.option_means, self.option_stds):
            action_dist_mu, action_dist_logstd = sess.run([mean, std], feed_dict={self.obs: obs})
            info.update({"action_dist_mu_o%d"%i : action_dist_mu, "action_dist_logstd_o%d"%i : action_dist_logstd })
            i += 1
        return act.ravel(), info

    def create_dupe(self):
        self.num_dupes += 1
        return GatedGaussianMLPPolicy(self.env,
                                      self.hidden_sizes,
                                      activation=self.activation,
                                      scope="policy_dupe_%d" % self.num_dupes,
                                      gate_hidden_sizes=self.gate_hidden_sizes,
                                      num_options=self.num_options,
                                      min_std = self.min_std,
                                      init_std = self.init_std,
                                      use_rms_filter = self.use_rms_filter,
                                      use_shared_layer = self.use_shared_layer)
