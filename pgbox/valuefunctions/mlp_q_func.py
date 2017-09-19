import tensorflow as tf
import numpy as np
from pgbox.utils import *
from pgbox.tf_utils import *
from pgbox.trpo.filters.tf_rms import *


def normalize_stats(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize_stats(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

class MLPQValueFunction(object):
    coeffs = None
    def __init__(self,
                 env,
                 hidden_sizes=(64,64),
                 activation=tf.nn.relu,
                 scope="qf",
                 max_iters = 15,
                 mixfrac = .1,
                 ln=False,
                 bn=True, # Note: if using batch norm, need to do: https://stackoverflow.com/documentation/tensorflow/7909/using-batch-normalization#t=201611300538141458755
                 use_old_ys=True,
                 use_rms_filter=True,
                 shared_rms_filter=None,
                 reward_scaling =None):
        self.net = None
        self.reward_scaling = reward_scaling
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.max_iters = max_iters
        self.use_old_ys = use_old_ys
        self.env = env
        self.timestep_limit = env.spec.timestep_limit
        self.observation_size = env.observation_space.shape[0]
        self.scope = scope
        self.action_size = env.action_space.shape[0]
        self.mixfrac = mixfrac
        self.use_rms_filter = use_rms_filter
        self.num_dupes = 0
        self.ln = ln
        self.bn = bn
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size], name="qf_obs")
        self.actions = tf.placeholder(tf.float32, [None, self.action_size], name="qf_act")
        self.y = tf.placeholder(tf.float32, [None, 1], name="qf_y")
        if use_old_ys:
            self.old_y = tf.placeholder(tf.float32, [None, 1], name="oldqf_y")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        if use_rms_filter:
            if shared_rms_filter is None:
                with tf.variable_scope(scope + "/obfilter"):
                    self.ob_rms = obs_rms = RunningMeanStd(shape=env.observation_space.shape)
            else:
                self.shared_obs_rms = obs_rms = shared_rms_filter

            net = tf.clip_by_value((self.obs - obs_rms.mean) / obs_rms.std, -5.0, 5.0)
        else:
            net = self.obs
        self.obs_input = net

        self.net = self._make_net(scope, activation, hidden_sizes, net, self.actions)

        if self.reward_scaling is not None:
            self.normalized_net = self.net
            self.net = self.reward_scaling.unnormalize_output(self.net)

        self.critic_target = self.y

        self.output = self.net
        self.placeholders = {}
        self.assigns = []

        for var in self.get_params(trainable=True):
            self.placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            self.assigns.append(tf.assign(var,self.placeholders[var.name]))

    def create_dupe(self):
        self.num_dupes += 1
        return MLPQValueFunction(self.env,
                                 hidden_sizes=self.hidden_sizes,
                                 activation=self.activation,
                                 scope="qf%d"%self.num_dupes,
                                 max_iters = self.max_iters,
                                 mixfrac = self.mixfrac,
                                 ln = self.ln,
                                 bn = self.bn,
                                 use_rms_filter=self.use_rms_filter,
                                 shared_rms_filter=self.shared_obs_rms,
                                 use_old_ys=self.use_old_ys,
                                reward_scaling =  self.reward_scaling)

    def filter_regularizable(self, var_list):
        var_list = [var for var in var_list if ('kernel' in var.name and not ('bias' in var.name) and not ('Adam' in var.name))]
        return var_list

    def filter_trainable(self, var_list):
        var_list = [var for var in var_list if (('kernel' in var.name or 'bias' in var.name or 'bn' in var.name or 'ln' in var.name) and not ('Adam' in var.name))]
        return var_list

    def output_vars(self):
        output_vars = [var for var in self.get_params(trainable=True) if 'outlayer' in var.name]
        return output_vars

    def get_params(self, trainable=False, regularizable=False):

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

        var_list = [var for var in var_list if not ('Adam' in var.name)]

        if trainable:
            var_list = self.filter_trainable(var_list)
        if regularizable:
            var_list = self.filter_regularizable(var_list)

        return var_list

    def set_param_values(self, sess, weights, trainable=False, regularizable=False):
        feed_dict = {}
        count = 0
        placeholders_needed = list(self.placeholders.keys())

        for var in self.get_params(trainable=trainable, regularizable=regularizable):
            # assert weights[count].shape == self.placeholders[var.name].get_shape().as_list()
            feed_dict[self.placeholders[var.name]] = weights[count]
            placeholders_needed.remove(var.name)
            count += 1

        if len(placeholders_needed) > 0:
            raise Exception("Missing placeholders: %s" % ' '.join(placeholders_needed))

        sess.run(self.assigns, feed_dict)

    def get_param_values(self, sess, trainable=False, regularizable=False):
        return sess.run(self.get_params(trainable=trainable, regularizable=regularizable))

    def _make_net(self, scope, activation, hidden_sizes, observations, actions, merge_layer=1, bn=False, ln=False, reuse=False, is_policy_opt=False):
        with tf.variable_scope(scope):
            net = observations
            # net = tf.layers.batch_normalization(net, name="qf_bnobservations", training=self.phase_train, reuse=reuse)
            # actions = tf.layers.batch_normalization(actions, name="qf_bnactions", training=self.phase_train, reuse=reuse)

            prev_out = net.get_shape().as_list()[-1]
            for i, x in enumerate(hidden_sizes):
                # In the low-dimensional case, we used batch
                # normalization on the state input and all layers of the µ network and all layers of the Q network prior
                # to the action input (details of the networks are given in the supplementary material:
                if merge_layer == i:
                    net = tf.concat([net, actions], axis=-1)

                if bn:
                    net = tf.layers.batch_normalization (net, training=self.phase_train, name="qf_bn%d"%i, reuse=reuse)
                elif ln:
                    net = tf.contrib.layers.layer_norm(net, center=True, scale=True, scope="qf_ln%d"%i, reuse=reuse)

                net = tf.layers.dense(inputs=net, units=x, activation=activation, kernel_initializer=tf.random_uniform_initializer(-1/np.sqrt(float(prev_out)), 1/np.sqrt(float(prev_out))), name="qf_h%d"%i, reuse=reuse)
                prev_out = x

            # Actions were not included until the 2nd hidden layer of Q.
            # otherwise we use the last layer to be a reward scaling layer...
            net = tf.layers.dense(inputs=net, units=1, activation=None, kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), name="qf_outlayer", reuse=reuse)

        return net

    def filter_regularizable(self, var_list):
        var_list = [var for var in var_list if ('kernel' in var.name and not ('bias' in var.name) and not ('Adam' in var.name))]
        return var_list

    def filter_trainable(self, var_list):
        var_list = [var for var in var_list if (('kernel' in var.name or 'bias' in var.name or 'ln' in var.name or 'bn' in var.name) and not ('Adam' in var.name))]
        return var_list

    def get_inputs(self, sess, observations, actions, ys, phase_train=True, infos={}):

        feed = {self.obs : observations, self.actions : actions, self.phase_train : phase_train, self.y : ys}
        if self.use_old_ys:
            yrpredold = self.get_qval(observations, actions, sess, normalized=True)
            yrpredold = yrpredold.reshape(-1, 1)
            feed.update({self.old_y : yrpredold})

        if self.reward_scaling:
            feed.update({self.reward_scaling.sigma_squared_input : (self.reward_scaling.sigma_squared(ys))})
        return feed

    def get_symbolic_with_trust_region(self, clip_param, action=None):
        if action is None:
            net = self.normalized_net
        else:
            net = self._make_net(self.scope, self.activation, self.hidden_sizes, self.obs_input, new_action_input, reuse=True, is_policy_opt=True)
        net = self.old_y + tf.clip_by_value(net - self.old_y, -clip_param, clip_param)

        if self.reward_scaling is not None:
            net = self.reward_scaling.unnormalize_output(net)

        return net

    def get_symbolic_with_action(self, new_action_input):
        """
        replaces the action input of the q function with a simple
        """
        return self._make_net(self.scope, self.activation, self.hidden_sizes, self.obs_input, new_action_input, reuse=True, is_policy_opt=True)

    def get_qval(self, next_obs, next_actions, sess, normalized=False):
        if normalized:
            ret = sess.run(self.normalized_net, feed_dict = {self.obs : next_obs, self.actions : next_actions, self.phase_train : False})[:,0]
        else:
            ret = sess.run(self.output, feed_dict = {self.obs : next_obs, self.actions : next_actions, self.phase_train : False})[:,0]
        return ret

    def predict(self, path, sess):
        ob_no = path["observations"]
        actions = path["actions"]
        ret = sess.run(self.output, feed_dict = {self.obs : ob_no, self.actions : actions, self.phase_train : False})[:,0]
        return ret
