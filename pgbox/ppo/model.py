

import multiprocessing
import os
import random
import time

import gym
import numpy as np
import tensorflow as tf

import pgbox.logging.logger as logger
from pgbox.policies.gated_gaussian_mlp_policy import *
from pgbox.tf_utils import *
from pgbox.threading_utils import *
from pgbox.trpo.rollouts import *
from pgbox.utils import *
from pgbox.valuefunctions.nn_vf import *

eps = 1e-8

class PPO(multiprocessing.Process):
    def __init__(self, args, observation_space, action_space, task_q, result_q, policy, vf,
                clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                optim_batchsize=64, learning_rate=3e-4, max_timesteps=10e7, schedule='constant', use_specialization=False):
        # TODO: get rid of args
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.schedule = schedule
        self.result_q = result_q
        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args
        self.vf = vf
        self.entcoeff = entcoeff
        self.policy = policy
        self.learning_rate = learning_rate
        self.max_timesteps = max_timesteps
        self.timesteps_so_far = 0
        self.optim_epochs = optim_epochs
        self.optim_batchsize = optim_batchsize
        self.use_specialization = use_specialization
        self.learning_rate_multiplier = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
        self.clip_param = clip_param * self.learning_rate_multiplier # Annealed cliping parameter epislon
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate * self.learning_rate_multiplier)

    def make_model(self):
        self.obs = self.policy.obs
        self.action = self.policy.action
        self.advantage = self.policy.advantage
        self.oldaction_dist_mu = self.policy.oldaction_dist_mu
        self.oldaction_dist_logstd = self.policy.oldaction_dist_logstd
        # means for each action
        self.action_dist_mu = self.policy.action_dist_mu
        # log standard deviations for each actions
        self.action_dist_logstd = self.policy.action_dist_logstd

        # TODO: this might not be necessary
        # batch_size = tf.shape(self.obs)[0]

        # what are the probabilities of taking self.action, given new and old distributions
        self.log_p_n = self.policy.log_p_n
        self.log_oldp_n = self.policy.log_oldp_n


        # kloldnew = oldpi.pd.kl(pi.pd)
        entropy = self.policy.ent
        mean_entropy = tf.reduce_mean(entropy)
        policy_entropy_penalty = (-self.entcoeff) * mean_entropy

        if isinstance(self.policy, GatedGaussianMLPPolicy) and self.use_specialization:
            surrs = []
            for i, option_lr in enumerate(self.policy.option_likelihood_ratio):
                surr1 = option_lr * self.policy.advantage * (self.policy.gate[:,i] + eps)
                surr2 = tf.clip_by_value(option_lr, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.policy.advantage * (self.policy.gate[:,i] + eps)
                surrs.append(tf.reshape(tf.minimum(surr1, surr2), [-1, 1]))

            combined_losses1 = tf.concat(surrs, axis=1)
            combined_losses1 = tf.reshape(tf.reduce_mean(combined_losses1, axis=1), [-1, 1])

            surr = - tf.reduce_mean(combined_losses1)
        else:
            surr1 = self.policy.likelihood_ratio * self.policy.advantage
            surr2 = tf.clip_by_value( self.policy.likelihood_ratio , 1.0 - self.clip_param, 1.0 + self.clip_param) * self.policy.advantage #
            surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

        if isinstance(self.vf, MLPConstrainedValueFunction):
            vfloss1 = tf.square(self.vf.output - self.vf.y)
            # TODO
            vpredclipped = self.vf.old_y + tf.clip_by_value(self.vf.output - self.vf.old_y, - self.clip_param, self.clip_param)
            vfloss2 = tf.square(vpredclipped - self.vf.y)
            vf_loss = tf.reduce_mean(tf.maximum(vfloss1, vfloss2)) # we do the same clipping-based trust region for the value function

            surr = .5 * surr + .5 * vf_loss + tf.reduce_mean(policy_entropy_penalty)
        else:
            surr = surr + tf.reduce_mean(policy_entropy_penalty)

        # TODO: may need to get current scope and then append to the policy scope
        var_list = self.policy.get_params()
        print("Varlist ", var_list)

        kl = self.policy.kl
        kl = tf.reduce_mean(kl)
        ent = self.policy.ent

        self.losses = [surr, kl, tf.reduce_mean(ent)]

        self.train_op = self.optimizer.minimize(surr)

        # the actual parameter values
        # call this to set parameter values
        initialize_uninitialized(self.session)
        # self.session.run(tf.global_variables_initializer())
        # value function
        # self.vf = VF(self.session)

        self.get_policy = GetPolicyWeights(self.session, var_list)

    def run(self):
        config = tf.ConfigProto(
            device_count = {'GPU': 0},
            #gpu_options = tf.GPUOptions(allow_growth=True)
        )
        self.session = tf.Session(config=config)
        self.make_model()

        while True:
            task = self.task_q.get()
            if task.code == LearnerTask.KILL_CODE:
                # TODO: self.terminate?
                self.task_q.cancel_join_thread()
                self.session.close()
                return
            elif task.code == LearnerTask.SET_EXTERNAL_POLICY_VALUES:
                weights = task.extra_params["weights"]
                print("Setting extrernal policy params learning task")
                self.policy.set_external_values(self.session, weights)
                time.sleep(0.1)
                self.task_q.task_done()
            elif task.code == LearnerTask.GET_PARAMS_CODE:
                self.task_q.task_done()
                self.result_q.put(LearnerResult(self.policy.get_param_values(self.session, optimizer_params=False)))
                print("Getting model %d params" % len(self.policy.get_param_values(self.session, optimizer_params=False)))
            elif task.code == LearnerTask.PUT_PARAMS_CODE:
                params = next_task.extra_params["weights"]
                logger.log("Setting model %d params" % len(params))
                # print(params)
                # the task is to set parameters of the actor policy
                self.policy.set_param_values(self.session, params)
                # super hacky method to make sure when we fill the queue with set parameter tasks,
                # an actor doesn't finish updating before the other actors can accept their own tasks.
                time.sleep(0.1)
                self.task_q.task_done()
            elif task.code == LearnerTask.ADJUST_MAX_KL:
                self.task_q.task_done()
                self.args.max_kl = task.extra_params['max_kl']
            elif task.code == LearnerTask.LEARN_PATHS:
                paths = task.extra_params['paths']
                stats = self.learn(paths)
                self.task_q.task_done()
                self.result_q.put(LearnerResult(self.policy.get_param_values(self.session, optimizer_params=False), stats))
            elif task.code == LearnerTask.REBUILD_NET:
                self.timesteps_so_far = 0 
                self.policy.rebuild_net(**task.extra_params)
                initialize_uninitialized(self.session)
                self.make_model()
                self.task_q.task_done()
                self.result_q.put(LearnerResult(self.policy.get_param_values(self.session, optimizer_params=False), stats))
            else:
                logger.log("Received unknown code! (%d)" % task.code)

        return

    def learn(self, paths):

        if self.schedule is 'constant':
            cur_lrmult = 1.0
            print("Using constant schedule")
        elif self.schedule is 'linear':
            cur_lrmult =  max(1.0 - float(self.timesteps_so_far) / self.max_timesteps, 0.01)
            print("Using linear schedule")
        elif self.schedule is 'quadratic':
            if self.timesteps_so_far > 0 and self.timesteps_so_far % 50e4 == 0:
                cur_lrmult *= .5
            else:
                cur_lrmult = 1.0
        else:
            raise NotImplementedError

        for path in paths:
            b = path["baseline"] = self.vf.predict(path, self.session)
            b1 = np.append(b, 0 if path["terminated"] else b[-1])
            deltas = path["rewards"] + self.args.gamma * b1[1:] - b1[:-1]
            path["advantage"] = discount(deltas, self.args.gamma * self.args.lam)
            path["returns"] = discount(path["rewards"], self.args.gamma)

        alladv = np.concatenate([path["advantage"] for path in paths])
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for path in paths:
            path["advantage"] = (path["advantage"] - mean) / (std + 1e-8)
        advant_n = np.concatenate([path["advantage"] for path in paths])

        # puts all the experiences in a matrix: total_timesteps x options
        # TODO: make this policy dependent like in rllab
        paths_concated = concat_tensor_dict_list(paths)

        action_dist_mu = paths_concated["info"]["action_dist_mu"]#np.concatenate([path["info"]["action_dist_mu"] for path in paths])
        action_dist_logstd = paths_concated["info"]["action_dist_logstd"]#np.concatenate([path["info"]["action_dist_logstd"] for path in paths])
        obs_n = paths_concated["observations"] #np.concatenate([path["observations"] for path in paths])
        action_n = paths_concated["actions"] #np.concatenate([path["actions"] for path in paths])

        # TODO: make this policy dependent like in rllab
        feed_dict = {self.obs: obs_n,
                     self.action: action_n,
                     self.advantage: advant_n,
                     self.oldaction_dist_mu: action_dist_mu,
                     self.oldaction_dist_logstd: action_dist_logstd}

        feed_dict.update(self.policy.get_extra_inputs(self.session, obs_n, paths_concated["info"]))
        feed_dict.update({self.learning_rate_multiplier : cur_lrmult})
        if isinstance(self.vf, MLPConstrainedValueFunction):
            feed_dict.update(self.vf.get_feed_vals(paths, self.session))

        if not isinstance(self.vf, MLPConstrainedValueFunction):
            self.vf.fit(paths, self.session)

        if hasattr(self.policy, "ob_rms") and not isinstance(self.policy, GatedGaussianMLPPolicy):
            # In the case of a GatedGaussian policy, we're going to share the gate/filter provided to us
            self.policy.ob_rms.update(obs_n, self.session)


        losses = []
        for _ in range(self.optim_epochs):
            # losses = [] # list of tuples, each of which gives the loss for a minibatch
            # for i in range(self.optim_epochs):
            # TODO: batchify
            _, newlosses = self.session.run([self.train_op, self.losses], feed_dict)
            losses.append(newlosses)

        surrogate_after, kl_after, entropy_after = self.session.run(self.losses, feed_dict)

        episoderewards = np.array([path["raw_rewards"].sum() for path in paths])
        realepisoderewards = np.array([path["rewards"].sum() for path in paths])
        stats = {}

        if "true_rewards" in paths[0]:
            truerewards = np.array([path["true_rewards"].sum() for path in paths])
            stats["TrueAverageReturn"] = truerewards.mean()
            stats["TrueStdReturn"] = truerewards.std()
            stats["TrueMaxReturn"] = truerewards.max()
            stats["TrueMinReturn"] = truerewards.min()

        logger.log("Min return agent_id: %d" % min(paths, key=lambda x: x["rewards"].sum())["agentid"])

        stats["ProcessedAverageReturn"] = realepisoderewards.mean()
        stats["ProcessedStdReturn"] = realepisoderewards.std()
        stats["ProcessedMaxReturn"] = realepisoderewards.max()
        stats["ProcessedMinReturn"] = realepisoderewards.min()
        baseline_paths = np.array([path['baseline'].sum() for path in paths])
        advantage_paths = np.array([path['advantage'].sum() for path in paths])
        stats["BaselineAverage"] = baseline_paths.mean()
        stats["AdvantageAverage"] = advantage_paths.mean()
        stats["Episodes"] = len(paths)
        stats["EpisodeAveLength"] = np.mean([len(path["rewards"]) for path in paths])
        stats["EpisodeStdLength"] = np.std([len(path["rewards"]) for path in paths])
        stats["EpisodeMinLength"] = np.min([len(path["rewards"]) for path in paths])
        stats["EpisodeMaxLength"] = np.max([len(path["rewards"]) for path in paths])
        stats["RawAverageReturn"] = episoderewards.mean()
        stats["RawStdReturn"] = episoderewards.std()
        stats["RawMaxReturn"] = episoderewards.max()
        stats["RawMinReturn"] = episoderewards.min()
        stats["Entropy"] = entropy_after
        stats["MaxKL"] = self.args.max_kl
        stats["Timesteps"] = sum([len(path["raw_rewards"]) for path in paths])
        # stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
        stats["KLDifference"] = kl_after
        stats["SurrogateLoss"] = surrogate_after
        # print ("\n********** Iteration {} ************".format(i))
        for k, v in sorted(stats.items()):
            logger.record_tabular(k,v)
        logger.dump_tabular()

        return stats
