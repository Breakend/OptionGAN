import argparse
import gym
import json
import multiprocessing
import numpy as np
import tensorflow as tf
import pgbox.logging.logger as logger
from pgbox.threading_utils import *
from pgbox.utils import *
from .trpo.model import *
from .ppo.model import *
from .trpo.rollouts import *

class ParallelAlgo(object):

    def __init__(self, algo_class, learner_env, policy, args, vf, init_params=None):
        self.task = type(learner_env.unwrapped).__name__
        args.max_pathlength = learner_env.spec.timestep_limit

        self.learner_tasks = multiprocessing.JoinableQueue()
        self.learner_results = multiprocessing.Queue()

        # TODO: extract policy from this to be separate
        self.learner = algo_class(args, learner_env.observation_space, learner_env.action_space, self.learner_tasks, self.learner_results, policy, vf)

        self.learner.start()
        self.iteration = 0

        self.rollouts = ParallelRollout(args, policy)

        if init_params is None:
            self.learner_tasks.put(GetParamsTask())
            self.learner_tasks.join()

            results = self.learner_results.get()
            self.starting_weights = results.policy
        else:
            self.learner_tasks.put(PutPolicyParamsTask(init_params))
            self.learner_tasks.join()

            self.starting_weights = init_params

        self.rollouts.set_policy_weights(self.starting_weights)

        self.start_time = time.time()

        # start it off with a big negative number
        self.last_reward = -1000000
        self.recent_total_reward = 0

        self.totalsteps = 0

        # self.n_steps = args.n_steps
        self.max_kl = args.max_kl
        self.timesteps_per_batch = args.timesteps_per_batch
        # self.decay_method = args.decay_method
        self.prev_mean_reward, self.prev_std_reward = 1e-8, 1e-8
        self.total_episodes = 0
        self.total_timesteps = 0
        self.args = args

    def get_policy_weights(self):
        return self.policy_weights

    def set_external_parameters(self, parameters):
        self.learner_tasks.put(SetExternalPolicyValuesTask(parameters))
        self.learner_tasks.join()
        self.rollouts.set_external_parameters(parameters)

    def step(self, paths=None, paths_processor = lambda x: x, train_steps=1):
        logger.log("................Starting iteration................")
        with logger.prefix('itr #%d | ' % self.iteration):
            # runs a bunch of async processes that collect rollouts
            logger.log("Iteration %d" % self.iteration)
            if paths is None:
                rollout_start = time.time()

                paths = self.rollouts.rollout()

                rollout_time = (time.time() - rollout_start) / 60.0

            self.total_episodes += len(paths)
            self.total_timesteps += sum([len(path["raw_rewards"]) for path in paths])
            logger.log("CumulativeEpisodes: %d" % self.total_episodes)
            logger.log("CumulativeTimesteps: %d" % self.total_timesteps)

            paths = paths_processor(paths)


            # Why is the learner in an async process?
            # Well, it turns out tensorflow has an issue: when there's a tf.Session in the main thread
            # and an async process creates another tf.Session, it will freeze up.
            # To solve this, we just make the learner's tf.Session in its own async process,
            # and wait until the learner's done before continuing the main thread.
            learn_start = time.time()
            for i in range(train_steps):
                self.learner_tasks.put(LearnFromPathsTask(paths))
                self.learner_tasks.join()
            results = self.learner_results.get()
            new_policy_weights, stats = results.policy, results.stats

            mean_reward = stats["RawAverageReturn"]
            std_reward = stats["RawStdReturn"]
            if "gate_dist" in paths[0]["info"]:
                gate_dists = []
                maxgate_dists = []
                mingate_dists = []
                for path in paths:
                    gate_dist = np.mean(path["info"]["gate_dist"], axis=0)
                    maxgate_dist = np.max(path["info"]["gate_dist"], axis=0)
                    mingate_dist = np.min(path["info"]["gate_dist"], axis=0)
                    gate_dists.append(gate_dist)
                    maxgate_dists.append(maxgate_dist)
                    mingate_dists.append(mingate_dist)
                gate_dists = np.vstack(gate_dists)
                logger.record_tabular("MeanGateDist", np.mean(gate_dists, axis=0))
                logger.record_tabular("MinGateDist", np.mean(mingate_dists, axis=0))
                logger.record_tabular("MaxGateDist", np.mean(maxgate_dists, axis=0))
                print(paths[0]["info"]["gate_dist"])

            learn_time = (time.time() - learn_start) / 60.0

            self.recent_total_reward += mean_reward

            logger.log("Total time: %.2f mins" % ((time.time() - self.start_time) / 60.0))
            logger.log("Current steps is " + str(self.timesteps_per_batch) + " and KL is " + str(self.max_kl))

            self.totalsteps += self.timesteps_per_batch
            self.prev_mean_reward = mean_reward
            self.prev_std_reward = std_reward
            self.iteration += 1
            logger.log("%d total steps have happened" % self.totalsteps)

            self.rollouts.set_policy_weights(new_policy_weights)
            self.policy_weights = new_policy_weights

            logger.dump_tabular(with_prefix=False)
        return self.iteration, paths

    def rebuild_net(self, **kwargs):
        self.learner_tasks.put(RebuildNetTask(**kwargs))
        self.learner_tasks.join()
        self.rollouts.rebuild_net(**kwargs)
        results = self.learner_results.get()
        self.rollouts.set_policy_weights(results.policy)

    def set_env(self, env):
        self.rollouts.set_env(env)

    def end(self):
        self.rollouts.end()
        self.learner.end()
        self.learner_results.close()
        self.learner_tasks.cancel_join_thread()
        self.learner_tasks.close()
        self.learner.terminate()

class ParallelTRPO(ParallelAlgo):

    def __init__(self, *args, **kwargs):
        ParallelAlgo.__init__(self, TRPO, *args, **kwargs)

class ParallelPPO(ParallelAlgo):

    def __init__(self, *args, **kwargs):
        ParallelAlgo.__init__(self, PPO,  *args, **kwargs)
