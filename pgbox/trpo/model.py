from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym
from pgbox.utils import *
from .rollouts import *
import time
import os
from pgbox.threading_utils import *

from pgbox.tf_utils import *
import random
import multiprocessing
import pgbox.logging.logger as logger
from pgbox.trpo.conjugate_gradient.hvp import *
from pgbox.policies.gated_gaussian_mlp_policy import GatedGaussianMLPPolicy
eps = 1e-8

class TRPO(multiprocessing.Process):
    def __init__(self, args, observation_space, action_space, task_q, result_q, policy, vf, use_specialization=False, add_sparsity_penalty=False):
        if use_specialization:
            args.max_kl *= policy.num_options
        multiprocessing.Process.__init__(self)

        self.task_q = task_q
        self.result_q = result_q
        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args
        self.vf = vf
        self.policy = policy
        self.use_specialization = use_specialization
        self.add_sparsity_penalties = add_sparsity_penalty

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

        print("Making surr")
        if isinstance(self.policy, GatedGaussianMLPPolicy) and self.use_specialization:
            print("DIMS CATTED")
            print(tf.concat([tf.reshape(x, [-1,1]) for x in self.policy.option_likelihood_ratio], axis=1))
            combined_options = tf.concat([tf.reshape(x, [-1,1]) for x in self.policy.option_likelihood_ratio], axis=1) * tf.reshape(self.policy.advantage, [-1,1])
            likelihood_ratio = tf.reshape(tf.reduce_sum(combined_options * self.policy.gate, axis=1), [-1])
            surr = likelihood_ratio
            surr = -tf.reduce_mean(surr)
        else:
            surr = - tf.reduce_mean(self.policy.likelihood_ratio * self.policy.advantage)


        if self.add_sparsity_penalties:
            self.lambda_s_loss = tf.constant(0.0)
            tau = .5

            gate = self.policy.gate
            self.lambda_s_loss = 10.0 * ( tf.reduce_mean((tf.reduce_mean(gate, axis=1) - (1.0 / self.policy.num_options))**2.))
            surr += self.lambda_s_loss

            self.lambda_v_loss = tf.constant(0.0)

            mean0, var0 = tf.nn.moments(gate, axes=[0])
            mean, var1 = tf.nn.moments(gate, axes=[1])
            self.lambda_v_loss = - 1.0 * (tf.reduce_mean(var0) + tf.reduce_mean(var1))
            surr += self.lambda_v_loss

        var_list = self.policy.get_params()
        print("Varlist ", var_list)

        if isinstance(self.policy, GatedGaussianMLPPolicy) and self.use_specialization:
            kl_firstfixeds = tf.stack(self.policy.option_kl_firstfixed)
            kls = tf.stack(self.policy.option_kl)
            kl = tf.reduce_sum(kls, axis=0)
            kl = tf.reduce_mean(kl)
            print("KLKLKL")
            print(kl_firstfixeds)
            kl_firstfixed = tf.reduce_mean(kl_firstfixeds, axis=0)
            kl_firstfixed = tf.reduce_mean(kl_firstfixed)
        else:
            kl_firstfixed = self.policy.kl_firstfixed
            kl_firstfixed = tf.reduce_mean(kl_firstfixed)

            kl = self.policy.kl
            kl = tf.reduce_mean(kl)

        ent = self.policy.ent # not gonna bother withe entropy for now

        self.losses = [surr, kl, tf.reduce_mean(ent)]

        # policy gradient
        self.pg = flatgrad(surr, var_list)

        self.hvp_func = build_fisher_hvp(kl_firstfixed, var_list, cg_damping = self.args.cg_damping)

        # the actual parameter values
        self.gf = GetFlat(self.session, var_list)
        # call this to set parameter values
        self.sff = SetFromFlat(self.session, var_list)
        initialize_uninitialized(self.session)
        # self.session.run(tf.global_variables_initializer())
        # value function
        # self.vf = VF(self.session)

        self.last_theta = self.gf()

    def run(self):
        config = tf.ConfigProto(
            device_count = {'GPU': 0},
            gpu_options = tf.GPUOptions(allow_growth=True)
        )

        self.session = tf.Session(config=config)

        print("Learner running!")
        self.make_model()
        print("Made model!")

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
                self.result_q.put(LearnerResult(self.policy.get_param_values(self.session)))
                print("Getting model %d params" % len(self.policy.get_param_values(self.session)))
            elif task.code == LearnerTask.PUT_PARAMS_CODE:
                params = task.extra_params["weights"]
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
                self.result_q.put(LearnerResult(self.policy.get_param_values(self.session), stats))
            elif task.code == LearnerTask.REBUILD_NET:
                print("Rebuilding learnernet with args: ", task.extra_params)
                self.policy.rebuild_net(**task.extra_params)
                self.add_sparsity_penalties = True
                self.make_model()
                initialize_uninitialized(self.session)
                self.task_q.task_done()
                self.result_q.put(LearnerResult(self.policy.get_param_values(self.session), stats))
            else:
                logger.log("Received unknown code! (%d)" % task.code)
        return

    def learn(self, paths):
        # is it possible to replace A(s,a) with Q(s,a)?
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


        # train value function / baseline on rollout paths
        self.vf.fit(paths, self.session)
        if hasattr(self.policy, "ob_rms") and not isinstance(self.policy, GatedGaussianMLPPolicy):
            # In the case of a GatedGaussian policy, we're going to share the gate/filter provided to us
            self.policy.ob_rms.update(obs_n, self.session)

        # TODO: make this policy dependent like in rllab
        feed_dict = {self.obs: obs_n,
                     self.action: action_n,
                     self.advantage: advant_n,
                     self.oldaction_dist_mu: action_dist_mu,
                     self.oldaction_dist_logstd: action_dist_logstd}

        feed_dict.update(self.policy.get_extra_inputs(self.session, obs_n, paths_concated["info"]))

        loss_before, kl_before, entropy_before = self.session.run(self.losses, feed_dict)


        logger.log("loss_before, kl_before, ent_before : %f,%f,%f" % (loss_before, kl_before, np.mean(entropy_before)))

        # parameters
        thprev = self.gf()

        fisher_vector_product = lambda x : self.hvp_func(x, session=self.session, feed_dict = feed_dict)

        g = self.session.run(self.pg, feed_dict)

        if np.allclose(g, 0):
            print("got zero gradient. not updating")
            return {}

        # solve Ax = g, where A is Fisher information metrix and g is gradient of parameters
        # stepdir = A_inverse * g = x
        stepdir = conjugate_gradient(fisher_vector_product, g)

        # let stepdir =  change in theta / direction that theta changes in
        # KL divergence approximated by 0.5 x stepdir_transpose * [Fisher Information Matrix] * stepdir
        # where the [Fisher Information Matrix] acts like a metric
        # ([Fisher Information Matrix] * stepdir) is computed using the function,
        # and then stepdir * [above] is computed manually.

        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))

        assert shs > 0

        lm = np.sqrt(shs / self.args.max_kl)

        fullstep = stepdir / lm

        logger.log("lagrange multiplier: %f gnorm: %f" % (lm, np.linalg.norm(g) ))
        def loss(th):
            self.sff(th)
            # surrogate loss: policy gradient loss
            return self.session.run(self.losses[:2], feed_dict)

        theta = linesearch2(loss, thprev, fullstep, self.args.max_kl) #negative_g_dot_steppdir / lm)
        self.sff(theta)

        surrogate_after, kl_after, entropy_after = self.session.run(self.losses,feed_dict)
        # print("new", self.session.run(self.action_dist_mu, feed_dict))

        if kl_after >= self.args.max_kl:
            logger.log("Violated KL constraint, rejecting step! KL-After (%f)" % kl_after)
            self.sff(thprev)
        if np.isnan(surrogate_after) or np.isnan(kl_after):
            logger.log("Violated because loss or KL is NaN")
            self.sff(thprev)
        if loss_before <= surrogate_after:
            logger.log("Violated because loss not improving... Prev (%f) After (%f)" % (loss_before, surrogate_after))
            self.sff(thprev)

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


    def end(self):
        self.task_q.put(KillThreadTask())
        time.sleep(0.25)
