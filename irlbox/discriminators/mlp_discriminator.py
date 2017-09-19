import tensorflow as tf
from pgbox.tf_utils import initialize_uninitialized, logit_bernoulli_entropy
from irlbox.nn_utils import Standardizer
import pgbox.logging.logger as logger
from pgbox.trpo.filters.tf_rms import *

class MLPDiscriminator(object):
    """
    Works well as of 7/12 able to reproduce original GAIL paper
    """

    def __init__(self,
                 observation_size,
                 hidden_sizes=(400,300),
                 activation=tf.nn.tanh,
                 learning_rate=1e-4,
                 scope="discriminator",
                 normalize_obs=False,
                 ent_reg_weight=0.0,
                 gradient_penalty_weight=0.0,
                 l2_penalty_weight=0.001,
                 objective="regular",
                 use_rms_filter=False,
                 num_epochs_per_step=3):
        self.observation_size = observation_size
        # self.target_size = target_size
        self.normalize_obs = normalize_obs
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.ent_reg_weight = ent_reg_weight
        self.num_epochs_per_step = num_epochs_per_step

        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )

        self.session = tf.Session(config=config)

        with tf.variable_scope(scope):
            if use_rms_filter:
                with tf.variable_scope("obfilter"):
                    self.ob_rms = RunningMeanStd(shape=(observation_size,))

                net_input = tf.clip_by_value((self.obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            else:
                net_input = self.obs

            net = net_input
            for i, x in enumerate(hidden_sizes):
                net = tf.layers.dense(inputs=net, units=x, activation=activation, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="discriminator_h%d"%i)
            net = tf.layers.dense(inputs=net, units=1, activation=None, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="discriminator_outlayer")

        self.net_out = net
        # TODO: maybe make this non-deterministic? MC dropout and then use a normal distribution?
        # action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, self.action_size)).astype(np.float32), name="policy_logstd")

        # loss function
        self.learning_rate = learning_rate

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        # Possible clipping from WGAN
        clip_ops = []
        for var in var_list:
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        self.clip_disc_weights_op = tf.group(*clip_ops)

        self.pred = tf.sigmoid(self.net_out) #-tf.log(1.-tf.sigmoid(self.net_out))
        self.reward = tf.sigmoid(self.net_out)#-tf.log(1.-tf.sigmoid(self.net_out))

        num_experts = tf.cast(tf.count_nonzero(self.targets), tf.int32)
        batch_size = tf.shape(self.obs)[0]

        #weights_B = tf.zeros(tf.shape(self.targets), tf.float32)
        #weights_bexp = weights_B[-num_experts:] + 1.0/(tf.cast(num_experts, tf.float32))
        #weights_bnovice = weights_B[:-num_experts] + 1.0/(tf.cast(batch_size - num_experts, tf.float32))
        #weights_B = tf.concat([weights_bnovice, weights_bexp], axis=0)

        if objective != "wgan":
            logger.log("Using sigmoid cross entropy discriminator objective")
            ent_B = logit_bernoulli_entropy(self.net_out)
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net_out[:-num_experts], labels=self.targets[:-num_experts]) - self.ent_reg_weight * ent_B[:-num_experts], axis=0)
            cross_entropy += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net_out[-num_experts:], labels=self.targets[-num_experts:]) - self.ent_reg_weight * ent_B[-num_experts:], axis=0)
            cross_entropy /= 2.

            #cross_entropy = tf.reduce_sum((cross_entropy - self.ent_reg_weight*ent_B)*weights_B, axis=0)
            self.loss = cross_entropy * 2.0 # reweighting to make focus on this instead of other penalty terms
        else:
            logger.log("Using wgan objective")
            disc_fake = self.net_out[:-num_experts]
            disc_real = self.net_out[-num_experts:]
            self.loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        self.l2_loss = tf.constant(0.0)

        if l2_penalty_weight > 0.0:
            loss_l2 = tf.add_n([ tf.nn.l2_loss(v) for v in var_list if 'kernel' in v.name and not ('Adam' in v.name)])/float(len(var_list)) * l2_penalty_weight
            self.l2_loss = loss_l2
            self.loss += loss_l2

        if gradient_penalty_weight > 0.0:
            batch_size = tf.shape(self.obs)[0]
            smallest = tf.minimum(num_experts, batch_size-num_experts)

            alpha = tf.random_uniform(
                shape=[smallest,1],
                minval=0.,
                maxval=1.
            )

            alpha_in = alpha*self.obs[-smallest:]
            beta_in = ((1-alpha)*self.obs[:smallest])
            interpolates = alpha_in + beta_in
            net2 = interpolates
            with tf.variable_scope(scope, reuse=True):
                for i, x in enumerate(hidden_sizes):
                    net2 = tf.layers.dense(inputs=net2, units=x, activation=tf.tanh, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="discriminator_h%d"%i)
                net2 = tf.layers.dense(inputs=net2, units=1, activation=None, kernel_initializer= tf.random_uniform_initializer(-0.05, 0.05), name="discriminator_outlayer")

            gradients = tf.gradients(net2, [interpolates])[0]
            gradients = tf.clip_by_value(gradients, -10., 10.)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = gradient_penalty_weight * tf.reduce_mean((slopes-1)**2)
            self.loss += gradient_penalty

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        comparison = tf.less(self.pred, tf.constant(0.5) )
        comparison2 = tf.less(self.targets, tf.constant(0.5) )
        overall = tf.cast(tf.equal(comparison, comparison2), tf.float32)
        accuracy = tf.reduce_mean(overall)#, tf.ones_like(self.targets)))
        accuracy_for_currpolicy = tf.reduce_mean(overall[:-num_experts])#, tf.ones_like(self.targets)))
        accuracy_for_expert = tf.reduce_mean(overall[-num_experts:])#, tf.ones_like(self.targets)))
        self.accuracy = accuracy
        self.accuracy_for_currpolicy = accuracy_for_currpolicy
        self.accuracy_for_expert = accuracy_for_expert


        # aux values
        # label_accuracy = tf.equal(tf.round(self.pred), tf.round(self.targets))
        # self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))
        # self.mse = tf.reduce_mean(tf.nn.l2_loss(self.pred - self.targets))
        # ones = tf.ones_like(self.targets)
        #
        # true_positives = tf.round(self.pred) * tf.round(self.targets)
        # predicted_positives = tf.round(self.pred)
        #
        # false_negatives = tf.logical_not(tf.logical_xor(tf.equal(tf.round(self.pred), ones), tf.equal(tf.round(self.targets), ones)))
        #
        # self.label_precision = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / tf.reduce_sum(tf.cast(predicted_positives, tf.float32))
        # self.label_recall = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / (tf.reduce_sum(tf.cast(true_positives, tf.float32)) + tf.reduce_sum(tf.cast(false_negatives, tf.float32)))

        initialize_uninitialized(self.session)

    def get_external_parameters(self):
        return None

    def step(self, observations, labels, aux_logging=True, clip_weights=False):
        if self.normalize_obs:
            self.inputnorm.update(observations)

        ops = [self.train_op, self.loss, self.accuracy, self.accuracy_for_expert, self.accuracy_for_currpolicy, self.l2_loss]

        for i in range(self.num_epochs_per_step):
            op_returns = self.session.run(ops, feed_dict={self.obs : observations, self.targets : labels})
            if clip_weights:
                self.session.run([self.clip_disc_weights_op])

        logger.log("Loss: %f" % op_returns[1])
        logger.log("LossL2: %f" % op_returns[5])
        logger.log("Accuracy: %f" % op_returns[2])
        logger.log("Accuracy (policy): %f" % op_returns[4])
        logger.log("Accuracy (expert): %f" % op_returns[3])


    def get_reward(self, batch):
        return self.session.run([self.reward], feed_dict={self.obs : batch})[0]
