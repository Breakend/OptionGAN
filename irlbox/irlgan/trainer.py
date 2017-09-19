import time
import numpy as np
from sklearn.utils import shuffle
from irlbox.utils import *
from pgbox.utils import get_filters
import pgbox.logging.logger as logger
import copy
class ParallelTrainer(object):

    def __init__(self, algo, expert_rollouts, discriminator, policy, args):
        expert_rollouts_tensor = np.vstack([step for path in expert_rollouts for step in path["observations"]])
        self.expert_rollouts_tensor = expert_rollouts_tensor

        self.discriminator = discriminator
        self.algo = algo
        self.iterations = 0
        self.args = args
        self.inverse = True

    def set_expert_rollouts(self, expert_rollouts):
        expert_rollouts_tensor = np.vstack([step for path in expert_rollouts for step in path["observations"]])
        self.expert_rollouts_tensor = expert_rollouts_tensor

    def set_algo(self, algo):
        self.algo = algo

    def stop_inverse(self):
        self.inverse = False
        self.is_first_noninverse = True

    def step(self):
        disc_process = time.time()
        mins, maxs, aves, stds = [],[],[],[]
        if self.inverse:
            logger.log("Training policy and extracting rewards...")
            itera, paths = self.algo.step(paths_processor = lambda x : process_samples_with_reward_extractor(x, self.discriminator, batch_size=50000))
        else:
            num_updates = 1
            if self.is_first_noninverse:
                num_updates = 1 
                self.is_first_noninverse = False
            itera, paths = self.algo.step(train_steps=num_updates)
            self.iterations += 1
            return self.iterations

        # unroll and stack novice observations
        logger.log("Processing rollouts for discriminator....")
        disc_process = time.time()
        observations = [step for path in paths for step in path["observations"]]
        observations = np.vstack(observations)
        novice_section = len(observations)

        # subsample experts according to size of observations
        #idx = np.random.randint(len(self.expert_rollouts_tensor), size=novice_section)
        useful_expert_rollouts = self.expert_rollouts_tensor#[idx, :]

        observations = np.concatenate([observations, useful_expert_rollouts], axis=0)
        if hasattr(self.discriminator, "ob_rms"): self.discriminator.ob_rms.update(observations, self.discriminator.session) # update running mean/std for policy

        labels = np.zeros((len(observations),))
        labels[novice_section:] = 1.0
        labels = labels.reshape((-1, 1))

        #observations, labels = shuffle(observations, labels)
        disc_proce_time = (time.time() - disc_process) / 60.0
        logger.log("Processed rollouts for disc in %f ...." % disc_proce_time)

        # TODO: merge rollouts with experts and add labels
        logger.log("Updating disc with %d rollouts" % novice_section)
        disc_process = time.time()
        self.discriminator.step(observations, labels)
        disc_proce_time = (time.time() - disc_process) / 60.0
        logger.log("Updated disc in %f ...." % disc_proce_time)

        external_parameters = self.discriminator.get_external_parameters()

        if external_parameters is not None: 
            self.algo.set_external_parameters(external_parameters)

        self.iterations += 1
        return self.iterations

    def end(self):
        self.algo.end()
