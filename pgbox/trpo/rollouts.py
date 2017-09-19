from __future__ import print_function, division
from builtins import range
import argparse
import gym
try:
    import gym_extensions
except:
    print("Couldn't import gym_extensions, can't use wrappers for envs")
import multiprocessing
import numpy as np
import tensorflow as tf
import time
import copy
from random import randint
import pgbox.logging.logger as logger
import threading
from pgbox.threading_utils import *

from pgbox.utils import *
from pgbox.tf_utils import *



class Actor(multiprocessing.Process):
    def __init__(self, args, task_q, result_q, actor_id, monitor, policy):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.agent_id = actor_id
        self.result_q = result_q
        self.args = args
        self.monitor = monitor
        self.policy = policy.create_dupe()

    def run(self):
        self.env = gym.make(self.args.task)

        if self.args.transformers:
            self.env = gym_extensions.wrappers.observation_transform_wrapper.ObservationTransformWrapper(
                self.env, self.args.transformers)

        self.env.seed(randint(0,999999))
        if self.monitor:
            self.env.monitor.start('monitor/', force=True)

        # self.observation_filter, self.reward_filter = get_filters(self.args, self.env.observation_space)

        # tensorflow variables (same as in model.py)
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = np.prod(self.env.action_space.shape)

        # tensorflow model of the policy
        self.obs = self.policy.obs
        self.debug = tf.constant([2,2])
        # self.action_dist_mu = self.policy.action_dist_mu
        # self.action_dist_logstd = self.policy.action_dist_logstd

        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        self.session = tf.Session(config=config)
        initialize_uninitialized(self.session)
        # self.session.run(tf.global_variables_initializer())

        while True:
            # get a task, or wait until it gets one
            next_task = self.task_q.get(block=True)

            if next_task.code == SamplingTask.COLLECT_SAMPLES_CODE:
                # the taskprint is an actor request to collect experience
                path = self.rollout()
                self.task_q.task_done()
                self.result_q.put(SamplingResult(path))
            elif next_task.code == SamplingTask.SET_EXTERNAL_POLICY_VALUES:
                weights = next_task.extra_params["weights"]
                print("Setting extrernal policy params")
                self.policy.set_external_values(self.session, weights)
                time.sleep(0.1)
                self.task_q.task_done()
            elif next_task.code == SamplingTask.KILL_CODE:
                logger.log("kill message")
                if self.monitor:
                    self.env.monitor.close()
                self.task_q.task_done()
                return
            elif next_task.code == SamplingTask.SET_ENV_TASK:
                env_name = next_task.extra_params["env"]
                print("setting new env! %s" %env_name)
                self.set_env(env_name)
                time.sleep(0.2)
                self.task_q.task_done()
            elif next_task.code == SamplingTask.PUT_PARAMS_CODE:
                params = next_task.extra_params["policy"]
                logger.log("Setting model %d params" % len(params))
                # print(params)
                # the task is to set parameters of the actor policy
                print("Setting rollout policy values of", [x.name for x in self.policy.get_params()])
                self.policy.set_param_values(self.session, params)
                # super hacky method to make sure when we fill the queue with set parameter tasks,
                # an actor doesn't finish updating before the other actors can accept their own tasks.
                time.sleep(0.1)
                self.task_q.task_done()
            elif next_task.code == SamplingTask.REBUILD_NET:
                print("Rebuilding net with args: ", next_task.extra_params)
                self.policy.rebuild_net(**next_task.extra_params)
                initialize_uninitialized(self.session)
                time.sleep(0.2)
                self.task_q.task_done()
            else:
                logger.log("Rollout thread got unknown task...")
        logger.log("Rollout thread dying")
        return

    def set_env(self, new_env):
        self.env = gym.make(new_env)

        if self.args.transformers:
            self.env = gym_extensions.wrappers.observation_transform_wrapper.ObservationTransformWrapper(
                self.env, self.args.transformers)

        self.env.seed(randint(0,999999))
        if self.monitor:
            self.env.monitor.start('monitor/', force=True)

    def rollout(self):
        obs, actions, rewards, raw_rewards, infos = [], [], [], [], []
        ob = self.env.reset()
        for i in range(self.args.max_pathlength):
            obs.append(ob)
            action, info = self.policy.act(ob, self.session)
            actions.append(action)
            infos.append(info)
            ob, reward, done, info = self.env.step(action)
            raw_rewards.append(reward)
            rewards.append(reward)
            if done:
                break

        path = {"observations": np.concatenate(np.expand_dims(obs, 0)),
                "info": concat_tensor_dict_list(infos),
                "rewards": np.array(rewards),
                "raw_rewards": np.array(raw_rewards),
                "actions":  np.array(actions),
                "terminated" : done,
                "agentid" : self.agent_id}

        return path

class ParallelRollout(object):
    def __init__(self, args, policy):
        self.args = args
        self._lock = threading.Lock()

        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()

        self.actors = []
        self.actors.append(Actor(self.args, self.tasks, self.results, 9999, args.monitor, policy))

        for i in range(self.args.num_threads-1):
            self.actors.append(Actor(self.args, self.tasks, self.results, 37*(i+3), False, policy))

        for a in self.actors:
            a.start()

        # we will start by running 20,000 / 1000 = 20 episodes for the first ieration

        self.average_timesteps_in_episode = 1000

    def rollout(self):

        # keep 20,000 timesteps per update
        #TODO: change this so that each worker does X samples and returns them, doesn't make sense to do it this way?
        num_rollouts = int(self.args.timesteps_per_batch / self.average_timesteps_in_episode)

        for i in range(num_rollouts):
            # TODO: pass num_samples here
            self.tasks.put(RunRolloutTask())

        self.tasks.join()

        paths = []
        while num_rollouts:
            num_rollouts -= 1
            rollouts_result = self.results.get()
            paths.append(rollouts_result.path)

        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)
        return paths

    def set_policy_weights(self, parameters):
        for i in range(self.args.num_threads):
            self.tasks.put(SetParamsTask(parameters))
        self.tasks.join()

    def set_env(self, name):
        for i in range(self.args.num_threads):
            self.tasks.put(SetEnvTask(name))
        self.tasks.join()

    def rebuild_net(self, **kwargs):
        for i in range(self.args.num_threads):
            self.tasks.put(RebuildNetSampleTask(**kwargs))
        self.tasks.join()

    def set_external_parameters(self, parameters):
        for i in range(self.args.num_threads):
            self.tasks.put(SetExternalPolicyValuesTask(parameters))
        self.tasks.join()

    def end(self):
        for i in range(self.args.num_threads):
            self.tasks.put(KillThreadTask())

        time.sleep(0.25)

        with self._lock:
            for p in self.actors:
                if p.is_alive():
                    print("Terminating %s" % p)
                    p.terminate()
