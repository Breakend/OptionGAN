import tensorflow as tf
from pgbox.tf_utils import initialize_uninitialized, logit_bernoulli_entropy, log10
from irlbox.nn_utils import Standardizer
import itertools
import pgbox.logging.logger as logger
import numpy as np
from pgbox.trpo.filters.tf_rms import *
from pgbox.distributions.categorical import *


# TODO: bound shifting information by doing second derivative as in TRPO

def get_cv_penalty(termination_importance_values):
    print("Using CV penalty")
    mean, var = tf.nn.moments(termination_importance_values, axes=[0])
    cv = var/mean
    return tf.nn.l2_loss(cv)

def get_mutual_info_penalty(options):
    print("Using Mutual info penalty")
    combos = [item for idx, item in enumerate(itertools.combinations(range(len(options)), 2))]
    mi = tf.Variable(0, dtype=tf.float32)
    for (i,j) in combos:

        # cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.multiply(tf.log(tf.sigmoid(self.discriminator_options[i].discrimination_logits) + TINY), tf.sigmoid(self.discriminator_options[j].discrimination_logits)), 1))
        # ent = tf.reduce_mean(-tf.reduce_sum(tf.multiply(tf.log(tf.sigmoid(self.discriminator_options[i].discrimination_logits) + TINY), tf.sigmoid(self.discriminator_options[i].discrimination_logits)), 1))
        # # ent = tf.reduce_mean(-tf.reduce_sum(tf.multiply(discriminator_options[i].discrimination_logits, discriminator_options[i].discrimination_logits), 1))
        # mi += (cond_ent + ent)
        # As defined in equation (4) @ https://www.cs.bham.ac.uk/~xin/papers/IJHIS-03-009-yao-liu.pdf
        mean_i, var_i = tf.nn.moments(options[i], axes=[0])
        mean_j, var_j = tf.nn.moments(options[j], axes=[0])
        mean_ij, var_ij = tf.nn.moments(tf.multiply(options[i], options[j]), axes=[0])
        # TODO: ^ Does this make sense mathematically ??
        corr_numerator = mean_ij-mean_i*mean_j
        corr_denominator = tf.square(var_i)*tf.square(var_j) + 1e-8
        corr_coeff = corr_numerator/corr_denominator
        mutual_info = -(1./2.0) * log10(1.0-tf.square(corr_coeff))

        mi += mutual_info
    mi /= float(len(combos))
    return tf.nn.l2_loss(mi)

class OptionatedMLPDiscriminator(object):

    def __init__(self,
                 observation_size,
                 hidden_sizes=(64,64),
                 gate_hidden_sizes=(128,128),
                 activation=tf.nn.tanh,
                 learning_rate=1e-4,
                 scope="discriminator",
                 normalize_obs=False,
                 ent_reg_weight=0.0,
                 gradient_penalty_weight=0.0,
                 l2_penalty_weight=0.0001,
                 mutual_info_penalty_weight = 0.01,
                 cv_penalty_weight = 0.0,
                 objective="regular",
                 num_epochs_per_step=2,
                 num_options = 2,
                 lambda_s = 10,
                 lambda_v = 1,
                 cross_entropy_reweighting=2.0,
                 use_rms_filter=False,
                 use_gated_trust_region=False,
                 use_shared_layer=True,
                 gate_change_penalty=.1,
                 uniform_distribution_rescale=1.0):
        self.observation_size = observation_size
        self.num_options = num_options
        self.hidden_sizes = hidden_sizes
        self.gate_hidden_sizes = gate_hidden_sizes
        self.activation = activation
        self.normalize_obs = normalize_obs
        self.use_gated_trust_region = use_gated_trust_region
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.ent_reg_weight = ent_reg_weight
        self.num_epochs_per_step = num_epochs_per_step
        self.scope = scope
        self.use_rms_filter = use_rms_filter
        self.orig_num_options = num_options
        self.use_shared_layer = use_shared_layer
        self.mutual_info_penalty_weight = mutual_info_penalty_weight
        self.cv_penalty_weight = cv_penalty_weight
        self.cross_entropy_reweighting = cross_entropy_reweighting
        self.l2_penalty_weight = l2_penalty_weight
        self.gate_change_penalty = gate_change_penalty
        self.lambda_s = lambda_s
        self.learning_rate = learning_rate
        self.uniform_distribution_rescale = uniform_distribution_rescale
        self.tau = .5# 1.0 / float(num_options)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.lambda_v = lambda_v
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )

        self.session = tf.Session(config=config)
        self.learning_rate = learning_rate

        self.options = []

        gating_network, net = self.build_network()

    def build_network(self, reuse=None, extra_options=0, stop_old_option_gradients=False):
        # build options
        scope = self.scope
        gate_hidden_sizes = self.gate_hidden_sizes
        hidden_sizes = self.hidden_sizes
        activation = self.activation
        num_options = self.num_options
        use_rms_filter = self.use_rms_filter
        use_gated_trust_region = self.use_gated_trust_region
        use_shared_layer = self.use_shared_layer
        l2_penalty_weight = self.l2_penalty_weight
        cv_penalty_weight = self.cv_penalty_weight
        mutual_info_penalty_weight = self.mutual_info_penalty_weight
        cross_entropy_reweighting = self.cross_entropy_reweighting
        gate_change_penalty = self.gate_change_penalty
        lambda_s = self.lambda_s
        lambda_v = self.lambda_v
        tau = self.tau
        self.options = []
        learning_rate = self.learning_rate
        with tf.variable_scope(scope):
            if use_rms_filter:
                if not hasattr(self, "ob_rms"):
                    with tf.variable_scope("obfilter"):
                        self.ob_rms = RunningMeanStd(shape=(self.observation_size,))

                net_input = tf.clip_by_value((self.obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            else:
                net_input = self.obs

            if use_shared_layer:
                shared_net = net_input
                for i, x in enumerate(hidden_sizes):
                    shared_net = tf.layers.dense(inputs=shared_net, units=x, activation=activation, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="discriminator_h%d"%i, reuse=reuse)
                for o in range(num_options):
                    with tf.variable_scope("option%d" % o):
                        net = tf.layers.dense(inputs=shared_net, units=1, activation=None, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="discriminator_outlayer", reuse=(reuse and o < num_options - extra_options))
                        if stop_old_option_gradients and o < num_options - extra_options:
                            net = tf.stop_gradient(net)
                        self.options.append(net)
            else:
                for o in range(num_options):
                    with tf.variable_scope("option%d" % o):
                        net = net_input
                        for i, x in enumerate(hidden_sizes):
                            net = tf.layers.dense(inputs=net, units=x, activation=activation, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="discriminator_h%d"%i, reuse=(reuse and o < num_options - extra_options))
                        net = tf.layers.dense(inputs=net, units=1, activation=None, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="discriminator_outlayer", reuse=(reuse and o < num_options - extra_options))
                        if stop_old_option_gradients and o < num_options - extra_options:
                            net = tf.stop_gradient(net)
                        self.options.append(net)

            # build gate
            with tf.variable_scope("gate"):
                gating_network = net_input
                for i, x in enumerate(gate_hidden_sizes):
                    gating_network = tf.layers.dense(inputs=gating_network, units=x, activation=activation, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="gating_hidden%d"%i, reuse=reuse)
                gating_network = tf.layers.dense(inputs=gating_network, units=num_options, activation=tf.nn.softmax, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="gating_outlayer_%d"%extra_options, reuse=False)
                self.termination_function = gating_network

            combined_options = tf.concat(self.options, axis=1)
            self.net_out = tf.reshape(tf.reduce_sum(combined_options * gating_network, axis=1), [-1, 1])

            self.termination_importance_values = tf.reduce_sum(self.termination_function, axis=0)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        self.pred = tf.sigmoid(self.net_out) #-tf.log(1.-tf.sigmoid(self.net_out))
        self.reward = tf.sigmoid(self.net_out)#-tf.log(1.-tf.sigmoid(self.net_out))

        num_experts = tf.cast(tf.count_nonzero(self.targets), tf.int32)
        batch_size = tf.shape(self.obs)[0]

        #weights_B = tf.zeros(tf.shape(self.targets), tf.float32)
        #weights_bexp = weights_B[-num_experts:] + 1.0/(tf.cast(num_experts, tf.float32))
        #weights_bnovice = weights_B[:-num_experts] + 1.0/(tf.cast(batch_size - num_experts, tf.float32))
        #weights_B = tf.concat([weights_bnovice, weights_bexp], axis=0)
        self.cross_entropy_loss = tf.constant(0.0)
        self.old_gating_output = tf.placeholder(tf.float32, [None, num_options])

        logger.log("Using sigmoid cross entropy discriminator objective")
        option_losses_exp = []
        option_losses_nov = []

        # http://www.cs.utoronto.ca/~fidler/teaching/2015/slides/CSC411/18_mixture.pdf
        for option in self.options:
            ent_B = logit_bernoulli_entropy(self.net_out)
            cross_entropy_nov = tf.nn.sigmoid_cross_entropy_with_logits(logits=option[:-num_experts], labels=self.targets[:-num_experts]) - self.ent_reg_weight * ent_B[:-num_experts]
            cross_entropy_exp = tf.nn.sigmoid_cross_entropy_with_logits(logits=option[-num_experts:], labels=self.targets[-num_experts:]) - self.ent_reg_weight * ent_B[-num_experts:]
            option_losses_exp.append(cross_entropy_exp)
            option_losses_nov.append(cross_entropy_nov)

        combined_losses_exp = tf.concat(option_losses_exp, axis=1)
        combined_losses_nov = tf.concat(option_losses_nov, axis=1)
        cross_entropy_nov = tf.reshape(tf.reduce_mean(combined_losses_nov * gating_network[:-num_experts], axis=1), [-1, 1])
        cross_entropy_exp = tf.reshape(tf.reduce_mean(combined_losses_exp * gating_network[-num_experts:], axis=1), [-1, 1])

        self.confidence = tf.sqrt(tf.reduce_mean(tf.square(self.old_gating_output - tf.reduce_mean(self.old_gating_output, axis=0))))
        clip_param = 1.0 - self.confidence # if very confident predictions generally, should clip changes very small.
        if use_gated_trust_region:

            clipped_gating_network = self.old_gating_output + tf.clip_by_value(gating_network - self.old_gating_output, - clip_param, clip_param)
            clipped_cross_entropy_nov = tf.reshape(tf.reduce_mean(combined_losses_nov * clipped_gating_network[:-num_experts], axis=1), [-1, 1])
            clipped_cross_entropy_exp = tf.reshape(tf.reduce_mean(combined_losses_exp * clipped_gating_network[-num_experts:], axis=1), [-1, 1])
            cross_entropy_nov = clipped_cross_entropy_nov#tf.maximum(clipped_cross_entropy_nov, cross_entropy_nov)
            cross_entropy_exp = clipped_cross_entropy_exp#tf.maximum(clipped_cross_entropy_exp, cross_entropy_exp)

        self.cross_entropy_loss = (tf.reduce_mean(cross_entropy_exp, axis=0) + tf.reduce_mean(cross_entropy_nov, axis=0)) / 2.0
        self.cross_entropy_loss *= cross_entropy_reweighting # reweight cross-entropy loss so other penalties don't affect it so much
        self.loss = self.cross_entropy_loss

        self.l2_loss = tf.constant(0.0)

        if l2_penalty_weight > 0.0:
            self.l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in var_list if 'kernel' in v.name and not ('Adam' in v.name)]) / float(len(var_list)) * l2_penalty_weight
            self.loss += self.l2_loss

        self.mi = tf.constant(0.0)

        if mutual_info_penalty_weight > 0.0:
            #TODO: fix mutual info so that each option is bounded here? maybe doesn't matter
            self.mi = get_mutual_info_penalty(self.options) * mutual_info_penalty_weight
            self.loss += self.mi

        self.cv = tf.constant(0.0)

        if cv_penalty_weight > 0.0:
            self.cv = get_cv_penalty(self.termination_importance_values) * cv_penalty_weight
            self.loss += self.cv

        self.gate_change = tf.constant(0.0)
        if gate_change_penalty > 0.0:
            self.gate_dist = Categorical(num_options)
            self.gate_change = gate_change_penalty * tf.reduce_mean(self.gate_dist.kl_sym(self.old_gating_output, gating_network))
            self.loss += self.gate_change


        # https://arxiv.org/abs/1511.06297
        # These two terms in ensemble encourage diversity and sparsity, while load balancing.
        # it's pretty amazing/awesome actually

        self.lambda_s_loss = tf.constant(0.0)

        if lambda_s > 0.0:
            gate = self.termination_function
            self.lambda_s_loss = lambda_s * (self.uniform_distribution_rescale * tf.reduce_mean((tf.reduce_mean(gate, axis=0) - tau)**2) +
                                    tf.reduce_mean((tf.reduce_mean(gate, axis=1) - tau)**2))
            self.loss += self.lambda_s_loss

        self.lambda_v_loss = tf.constant(0.0)

        if lambda_v > 0.0:
            gate = self.termination_function
            if use_gated_trust_region:
                gate = self.old_gating_output + tf.clip_by_value(self.termination_function - self.old_gating_output, - clip_param, clip_param)
            mean0, var0 = tf.nn.moments(gate, axes=[0])
            mean, var1 = tf.nn.moments(gate, axes=[1])
            self.lambda_v_loss = - lambda_v * (tf.reduce_mean(var0) + tf.reduce_mean(var1))
            self.loss += self.lambda_v_loss

        self.train_op = self.optimizer.minimize(self.loss)
        comparison = tf.less(self.pred, tf.constant(0.5) )
        comparison2 = tf.less(self.targets, tf.constant(0.5) )
        overall = tf.cast(tf.equal(comparison, comparison2), tf.float32)
        accuracy = tf.reduce_mean(overall)#, tf.ones_like(self.targets)))
        accuracy_for_currpolicy = tf.reduce_mean(overall[:-num_experts])#, tf.ones_like(self.targets)))
        accuracy_for_expert = tf.reduce_mean(overall[-num_experts:])#, tf.ones_like(self.targets)))
        self.accuracy = accuracy
        self.accuracy_for_currpolicy = accuracy_for_currpolicy
        self.accuracy_for_expert = accuracy_for_expert

        initialize_uninitialized(self.session)
        return gating_network, net

    def rebuild_net(self, extra_options=0, stop_old_option_gradients=True):
        self.num_options += extra_options
        self.build_network(reuse=True, extra_options=extra_options, stop_old_option_gradients=stop_old_option_gradients)

    def get_external_parameters(self):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        var_list = [var for var in var_list if (('gate' in var.name or 'filter' in var.name) and not 'Adam' in var.name)]
        return zip([var.name for var in var_list], self.session.run(var_list))

    def step(self, observations, labels, aux_logging=True):
        if self.normalize_obs:
            self.inputnorm.update(observations)

        ops = [self.train_op,
               self.loss,
               self.accuracy,
               self.accuracy_for_expert,
               self.accuracy_for_currpolicy,
               self.mi,
               self.cv,
               self.lambda_s_loss,
               self.lambda_v_loss,
               self.termination_function,
               self.l2_loss,
               self.cross_entropy_loss,
               self.gate_change,
               self.confidence]

        old_gate_vals = self.session.run(self.termination_function, feed_dict={self.obs : observations, self.targets : labels})

        for i in range(self.num_epochs_per_step):
            op_returns = self.session.run(ops, feed_dict={self.obs : observations, self.targets : labels, self.old_gating_output : old_gate_vals})

        logger.log("Loss: %f" % op_returns[1])
        logger.log("Accuracy: %f" % op_returns[2])
        logger.log("Accuracy (policy): %f" % op_returns[4])
        logger.log("Accuracy (expert): %f" % op_returns[3])
        logger.log("MI: %f" % op_returns[5])
        logger.log("cv: %f" % op_returns[6])
        logger.log("l2_loss: %f" % op_returns[10])
        logger.log("ce_loss: %f" % op_returns[11])
        logger.log("gate_change: %f" % op_returns[12])
        logger.log("lambda_s_loss: %f" % op_returns[7])
        logger.log("lambda_v_loss: %f" % op_returns[8])
        logger.log("Importance: {}".format(str(np.mean(np.array(op_returns[9]), axis=0))))
        logger.log("Gate Confidence: %f" % op_returns[13])
        print(op_returns[9])


    def get_reward(self, batch):
        return self.session.run([self.reward], feed_dict={self.obs : batch})[0]
