# FROM: https://raw.githubusercontent.com/shaneshixiang/rllabplusplus/master/sandbox/rocky/tf/algos/ddpg.py
import pgbox.logging.logger as logger
import numpy as np
import pyprind
import time
import gc
import tensorflow as tf
from pgbox.tf_utils import *
from pgbox.utils import *
from pgbox.sampling_utils import rollout
from pgbox.sampling.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from pgbox.trpo.filters.tf_rms import *
from pgbox.util.schedules import LinearSchedule
import random

def select_optimizer(optimizer_method, lr):
    optimizer_map = {
        "adam" : tf.train.AdamOptimizer(lr),
        "sgd" : tf.train.GradientDescentOptimizer(lr)
        }
    return optimizer_map[optimizer_method]

class DDPG(object):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            es,
            batch_size=64,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            replacement_prob=1.0,
            discount=0.99,
            # max_path_length=250,
            qf_weight_decay=0.0,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0.0,
            policy_update_method='adam',
            policy_learning_rate=1e-4,
            policy_updates_ratio=1.0,
            eval_samples=10000,
            soft_target=False,
            soft_target_tau=0.001,
            target_network_update_freq=500,
            n_updates_per_sample=1,
            scale_reward=1.,
            include_horizon_terminal_transitions=False,
            use_q_trust_region_updates = False,
            use_policy_trust_region_updates = False,
            normalize_gradients = False,
            clip_param = .05,
            ob_rms = None,
            normalize_returns = False,
            reward_scaling=None,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
            prioritized_replay_beta0=0.4,
            prioritized_replay_beta_iters=None,
            prioritized_replay_eps=1e-6,
            keep_best = True,
            bayes_baseline = False,
            termination = None,
            likelihood_bayes=True,
            average_on_policy_updates=True,
            stochastic=True,
            **kwargs):
        """
        :param env: Environment
        :param policy: Policy
        :param qf: Q function
        :param es: Exploration strategy
        :param batch_size: Number of samples for each minibatch.
        :param n_epochs: Number of epochs. Policy will be evaluated after each epoch.
        :param epoch_length: How many timesteps for each epoch.
        :param min_pool_size: Minimum size of the pool to start training.
        :param replay_pool_size: Size of the experience replay pool.
        :param discount: Discount factor for the cumulative return.
        :param max_path_length: Discount factor for the cumulative return.
        :param qf_weight_decay: Weight decay factor for parameters of the Q function.
        :param qf_update_method: Online optimization method for training Q function.
        :param qf_learning_rate: Learning rate for training Q function.
        :param policy_weight_decay: Weight decay factor for parameters of the policy.
        :param policy_update_method: Online optimization method for training the policy.
        :param policy_learning_rate: Learning rate for training the policy.
        :param eval_samples: Number of samples (timesteps) for evaluating the policy.
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained
        :param scale_reward: The scaling factor applied to the rewards when training
        :param include_horizon_terminal_transitions: whether to include transitions with terminal=True because the
        horizon was reached. This might make the Q value back up less stable for certain tasks.
        :return:
        """
        self.env = env
        self.policy = policy
        self.qf = qf
        self.stochastic = stochastic
        self.es = es
        self.soft_target = soft_target
        self.reward_scaling = reward_scaling
        self.prioritized_replay=prioritized_replay
        self.prioritized_replay_alpha=prioritized_replay_alpha
        self.prioritized_replay_beta0=prioritized_replay_beta0
        self.prioritized_replay_beta_iters=prioritized_replay_beta_iters
        self.prioritized_replay_eps=prioritized_replay_eps
        self.dual_asynchronous_q = False
        self.bayes_baseline = bayes_baseline
        self.num_second_q_updates = 2
        self.termination = termination
        self.normalize_gradients = normalize_gradients
        self.normalize_returns = normalize_returns
        self.batch_size = batch_size
        self.average_on_policy_updates = average_on_policy_updates
        self.clip_param = clip_param
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.replacement_prob = replacement_prob
        self.discount_gamma = discount
        self.max_path_length = env.spec.timestep_limit
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = select_optimizer(qf_update_method, qf_learning_rate)
        self.qf_learning_rate = qf_learning_rate
        self.policy_weight_decay = policy_weight_decay
        self.policy_update_method = select_optimizer(policy_update_method, policy_learning_rate)
        self.policy_learning_rate = policy_learning_rate
        self.policy_updates_ratio = policy_updates_ratio
        self.eval_samples = eval_samples
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.target_network_update_freq = target_network_update_freq
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.use_q_trust_region_updates = use_q_trust_region_updates
        self.use_policy_trust_region_updates = use_policy_trust_region_updates
        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.likelihood_bayes = likelihood_bayes
        self.qf_l2_loss_averages = []
        self.qf2losses = []
        self.use_termination_prob = not (self.termination is None)
        self.qf2values = []

        self.policy_l2_loss_averages = []

        self.q_averages = []
        self.y_averages = []
        self.termination_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.scale_reward = scale_reward

        self.train_policy_itr = 0

        self.opt_info = None
        self.gc_dump_time = time.time()
        if self.use_termination_prob:
            self.termination_update_method = select_optimizer(qf_update_method, qf_learning_rate)

        if self.prioritized_replay:
            self.pool = PrioritizedReplayBuffer(self.replay_pool_size, alpha=prioritized_replay_alpha, keep_best=keep_best)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = n_epochs * epoch_length
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
        else:
            self.pool = ReplayBuffer(
                self.replay_pool_size,
                keep_best = keep_best
            )

        self.ob_rms = ob_rms

        config = tf.ConfigProto(
            device_count = {'GPU': 1},
            gpu_options = tf.GPUOptions(allow_growth=True)
        )
        self.sess = tf.Session(config=config)


        self.make_model()
        self.itr = 0
        self.epoch = 0
        # self.sample_policy = self.policy.create_dupe()

        observation = self.env.reset()
        self.total_timesteps = 0

        # self.sample_policy = self.policy.create_dupe()
        initialize_uninitialized(self.sess)

        # self.update_target_policy = self._update_target_func(self.policy, self.target_policy, tau=soft_target_tau)
        self.update_target_q= self._update_target_func(self.qf, self.target_qf, tau=soft_target_tau)
        # self.update_sample_policy = self._update_target_func(self.policy, self.sample_policy)

    def _update_target_func(self, q, q_target, tau = 1.):
        # TODO: soft update
        update_target_expr = []
        for var, var_target in zip(sorted(q.get_params(), key=lambda v: v.name),
                                   sorted(q_target.get_params(), key=lambda v: v.name)):
            update_target_expr.append(tf.assign(var_target, (1. - tau) * var_target + tau * var))

        update_target_expr = tf.group(*update_target_expr)
        return update_target_expr


    def step(self):
        """
        Step in this case counts as an epoch
        """
        itr = self.itr
        path_length = 0
        path_return = 0
        terminal = False
        initial = False
        self.epoch += 1
        observation = self.env.reset()

        logger.push_prefix('epoch #%d | ' % self.epoch)
        logger.log("Training started")
        train_qf_itr, train_policy_itr = 0, 0

        path = []
        for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
            # Execute policy
            if terminal:
                # print("terminal")
                # Note that if the last time step ends an episode, the very
                # last state and observation will be ignored and not added
                # to the replay pool
                observation = self.env.reset()
                # sample_policy.reset()
                self.es_path_returns.append(path_return)
                self.es.reset()
                path_length = 0
                path_return = 0
                initial = True
            else:
                initial = False


            action, _ = self.policy.act_sample(observation, self.sess)# self.es.act(itr, observation, sess=self.sess, policy=)  # qf=qf)

            next_observation, reward, terminal, _ = self.env.step(action)
            self.total_timesteps += 1
            path_length += 1
            path_return += reward

            if not terminal and path_length >= self.max_path_length:
                terminal = True
                # TODO: fix this, This is only true of tasks where ending early is bad
                failure = False
                # only include the terminal transition in this case if the flag was set
                if self.include_horizon_terminal_transitions:
                    self.pool.add(observation, action, reward * self.scale_reward, next_observation, terminal, failure)
            else:
                failure=True
                sample = (observation, action, reward * self.scale_reward, next_observation, terminal, failure)
                self.pool.add(*sample)
                path.append(sample)

            observation = next_observation

            if len(self.pool) >= self.min_pool_size:
                for update_itr in range(self.n_updates_per_sample):
                    # Train policy
                    if hasattr(self, "beta_schedule"):
                        batch = self.pool.sample(self.batch_size, beta=self.beta_schedule.value(self.total_timesteps))
                    else:
                        batch = self.pool.sample(self.batch_size)

                    itrs = self.do_training(itr, batch)

                    train_qf_itr += itrs[0]
                    train_policy_itr += itrs[1]
                # self.sess.run(self.update_sample_policy)

            itr += 1
            if time.time() - self.gc_dump_time > 100:
                gc.collect()
                self.gc_dump_time = time.time()

        logger.log("Training finished")
        logger.log("Trained qf %d steps, policy %d steps"%(train_qf_itr, train_policy_itr))
        rollouts = []
        if len(self.pool) >= self.min_pool_size:
            rollouts = self.evaluate(self.epoch)
        logger.dump_tabular(with_prefix=False)
        logger.pop_prefix()

        self.itr = itr
        return itr, rollouts

    def train_termination(self, observations, actions, ys, weights):
        feed_dict = self.termination.get_inputs(self.sess, observations, actions, ys)
        feed_dict.update(self.policy.get_inputs(self.sess, observations))
        # feed_dict.update({self.yvar : ys})
        if self.prioritized_replay:
            feed_dict.update({self.importance_weights : weights})
        return self.sess.run([self.termination_loss, self.termination.output, self.termination_training_func], feed_dict=feed_dict)

    def get_policy_loss(self, observations, actions):
        feed_dict = self.qf.get_inputs(self.sess, observations, actions)
        feed_dict.update(self.policy.get_inputs(self.sess, observations))

        if self.use_termination_prob:
            feed_dict.update(self.termination.get_inputs(self.sess, observations, actions, np.zeros((len(actions), 1))))

        return self.sess.run([self.policy_loss, self.policy_weight_decay_term], feed_dict=feed_dict)

    def train_policy(self, observations, actions):
        feed_dict = {}
        feed_dict.update({self.qf.obs : observations})#= self.qf.get_inputs(self.sess, observations, actions, np.zeros((len(actions), 1)))
        feed_dict.update(self.policy.get_inputs(self.sess, observations, actions))

        if self.use_termination_prob:
            feed_dict.update(self.termination.get_inputs(self.sess, observations, actions, np.zeros((len(actions), 1))))

        return self.sess.run([self.policy_loss, self.policy_weight_decay_term, self.policy_training_func], feed_dict=feed_dict)

    def get_monte_carlo_value(self, observations, actions):
        feed_dict = {}
        feed_dict.update({self.qf.obs : observations})#= self.qf.get_inputs(self.sess, observations, actions, np.zeros((len(actions), 1)))
        feed_dict.update(self.policy.get_inputs(self.sess, observations, actions))

        if self.use_termination_prob:
            feed_dict.update(self.termination.get_inputs(self.sess, observations, actions, np.zeros((len(actions), 1))))

        return self.sess.run([self.adv], feed_dict=feed_dict)[0]

    def make_model(self):
        if self.stochastic:
            #TODO: make a DDPG-style update, and then a TRPO style advantage update...
            # Here, our advantage is estimated online, but no gradients backprop through it
            pqval = self.qf.get_symbolic_with_action(self.policy.action)
            # stopping the gradient into the policy because this is onnly related to the Q update
            adv = self.policy.monte_carlo_average(self.qf.get_symbolic_with_action, stop_gradient=True)
            self.adv = adv
            advantage = pqval - adv #lambda x : self.qf.get_symbolic_with_action(x) #- adv
            # advantage = (pqval - adv)
            # TODO: this is somehow always just one...
            surr1 = self.policy.bayesianlikelihood_ratio * tf.stop_gradient(advantage) #TODO: idk if this bayesian likelihood thing is principled. this is the like two priors...
            surr2 = tf.clip_by_value( self.policy.bayesianlikelihood_ratio , 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        else:
            # Here, our gradients backprop through the estimated advantage function
            # pqval = self.qf.get_symbolic_with_action(self.policy.get_action_sym())
            adv = self.policy.monte_carlo_average(self.qf.get_symbolic_with_action, stop_gradient=False)
            self.adv = adv
            # TODO: this is somehow always just one...
            surr = - tf.reduce_mean(adv) # PPO's pessimistic surrogate (L^CLIP)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.policy_weight_decay_term = tf.contrib.layers.apply_regularization(self.policy.regularizer, reg_variables)
        self.policy_loss = surr + self.policy_weight_decay_term

        # This initializes the Q function update targets, here we're trying to update based on the advantage to give us a baseline
        # and more stable updates
        self.qf.init_optimizer(self.qf.y, adv)

        #  \sum_(for a in ) Qpi(s,a) - Vpi(s)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            policy_varlist = self.policy.get_params()
            grads = self.policy_update_method.compute_gradients(self.policy_loss, policy_varlist)
            grads = [(tf.clip_by_value(grad, -10.0, 10.0), var)
                          for grad, var in grads if grad is not None]
            # grads_and_vars = list(zip(grads, policy_varlist))
            self.policy_training_func = self.policy_update_method.apply_gradients(grads)

        self.target_qf = self.qf.create_dupe(scope="targetqf")
        # self.target_policy = self.policy.create_dupe()

        if self.use_termination_prob:
            self.termination_loss = tf.reduce_mean(tf.square(self.termination.y - self.termination.output))
            grads = self.termination_update_method.compute_gradients(self.termination_loss, self.termination.get_params())
            self.termination_training_func = self.termination_update_method.apply_gradients(grads)
            self.target_termination = self.termination.create_dupe("target_termination_func")
            self.update_target_termination = self._update_target_func(self.termination, self.target_termination, tau=self.soft_target_tau)

    def do_training(self, itr, batch, on_pol_batch = None):
        obs, actions, rewards, next_obs, terminals, failures, weights, idxs = batch
        if on_pol_batch:
            obs_onpol, actions_onpol, rewards_onpol, next_obs_onpol, terminals_onpol, failures_onpol, weights_onpol, idxs_onpol = batch

        self.ob_rms.update(obs, self.sess)

        if hasattr(self.es, "optim"):
            self.es.optimize(self.sess, obs, actions)

        # target_policy = self.target_policy
        target_qf = self.target_qf

        # TODO: this should be
        #  r + (1 - terminal) * V(s) approximated by dropout., then we minimize the error with this?

        # next_actions, _ = self.policy.get_actions(next_obs, self.sess)
        # next_qvals = target_qf.get_qval(next_obs, next_actions, self.sess)
        next_qvals = self.get_monte_carlo_value(next_obs, actions)

        ys = rewards + (1. - terminals) * self.discount_gamma * next_qvals.reshape(-1)

        if self.use_termination_prob:
            terminations = failures + self.discount_gamma * self.target_termination.get_qval(next_obs, next_actions, self.sess).reshape(-1)
            term_loss, term_val, _ =  self.train_termination(obs, actions, terminations.reshape(-1, 1), weights.reshape(-1, 1))
            self.sess.run(self.update_target_termination)
            self.termination_averages.append(term_val)

        extra_feed = self.policy.get_inputs(self.sess, obs, actions)
        td_error, qf_loss, qf_weight_decay_loss, qval, _  = self.qf.fit(obs, actions, ys.reshape(-1,1), self.sess, extra_feed)
        # td_error, qf_loss, qf_weight_decay_loss, qval, _ = self.train_qval(obs, actions, ys.reshape(-1, 1), weights.reshape(-1, 1))

        if self.prioritized_replay:
            new_priorities = np.abs(td_error) + self.prioritized_replay_eps
            self.pool.update_priorities(idxs, new_priorities)

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval.reshape(-1))
        self.y_averages.append(ys)
        self.qf_l2_loss_averages.append(qf_weight_decay_loss)

        self.train_policy_itr += self.policy_updates_ratio
        train_policy_itr = 0

        while self.train_policy_itr > 0:
            policy_surr, policy_l2_loss, _ = self.train_policy(obs, actions)
            # TODO: make this update line cleaner
            self.policy_surr_averages.append(policy_surr)
            self.policy_l2_loss_averages.append(policy_l2_loss)
            self.train_policy_itr -= 1
            train_policy_itr += 1

        # self.sess.run(self.update_target_policy)
        self.sess.run(self.update_target_q)

        return 1, train_policy_itr # number of itrs qf, policy are trained

    def evaluate(self, epoch):
        logger.log("Collecting samples for evaluation")

        num_samples = 0
        paths = []
        # import pdb; pdb.set_trace()

        while num_samples < self.eval_samples:
            path = rollout(self.env, self.policy, self.max_path_length, self.sess)
            num_samples += len(path["rewards"])
            paths.append(path)

        self.env.reset()


        returns = [np.sum(path["rewards"]) for path in paths]

        average_discounted_return = np.mean(
            [discount_return(path["rewards"], self.discount_gamma) for path in paths]
        )

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)
        average_policy_surr = np.mean(self.policy_surr_averages)
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        # policy_reg_param_norm = np.linalg.norm(self.policy.get_param_values(self.sess))
        # qfun_reg_param_norm = np.linalg.norm(self.qf.get_param_values(self.sess))

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Iteration', epoch)
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        if len(self.termination_averages) > 0:
            logger.record_tabular('TerminationVal', np.mean(self.termination_averages))
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
        if hasattr(self.es, 'get_and_clear_losses'):
            logger.record_tabular('ESLoss', self.es.get_and_clear_losses())
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AverageQL2Loss', np.mean(self.qf_l2_loss_averages))
        logger.record_tabular('AveragePolicySurr', average_policy_surr)
        logger.record_tabular('AveragePolicyL2Loss',np.mean(self.policy_l2_loss_averages))
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))
        if self.dual_asynchronous_q:
            all_q2s = np.concatenate(self.qf2values)
            logger.record_tabular('AverageQ2Loss', np.mean(self.qf2losses))
            logger.record_tabular('AverageQ2', np.mean(all_q2s))
            logger.record_tabular('AverageAbsQ2', np.mean(np.abs(all_q2s)))
        logger.record_tabular('AverageAction', average_action)
        logger.record_tabular('TotalTimesteps', self.total_timesteps)

        self.qf_loss_averages = []
        self.policy_surr_averages = []

        self.q_averages = []
        self.y_averages = []
        self.termination_averages = []
        self.es_path_returns = []
        return paths

    def end(self):
        return
