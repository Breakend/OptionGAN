import pickle
import numpy as np
import pgbox.logging.logger as logger
import time

def batchify_dict(samples, batch_size, total_len):
    for i in range(0, total_len, batch_size):
        yield select_from_tensor_dict(samples, i, min(total_len, i+batch_size))

def batchify_list(samples, batch_size):
    total_len = len(samples)
    for i in range(0, total_len, batch_size):
        yield select_from_tensor(samples, i, min(total_len, i+batch_size))

def select_from_tensor_dict(tensor_dict, start, end):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = select_from_tensor_dict(tensor_dict[k], start, end)
        else:
            ret[k] = select_from_tensor(tensor_dict[k], start, end)
    return ret

def select_from_tensor(x, start, end):
    return x[start:end]


def shorten_tensor_dict(tensor_dict, max_len):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = shorten_tensor_dict(tensor_dict[k], max_len)
        else:
            ret[k] = shorten_tensor(tensor_dict[k], max_len)
    return ret

def shorten_tensor(x, max_len):
    return x[:max_len]

def load_expert_rollouts(filepath, max_traj_len = -1, num_expert_rollouts = 10):
    # why encoding? http://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    expert_rollouts = pickle.load(open(filepath, "rb"), encoding='latin1')

    # In the case that we only have one expert rollout in the file
    if type(expert_rollouts) is dict:
        expert_rollouts = [expert_rollouts]

    expert_rollouts = expert_rollouts[:min(len(expert_rollouts), num_expert_rollouts)]

    if max_traj_len > 0:
        expert_rollouts = [shorten_tensor_dict(x, traj_len) for x in expert_rollouts]

    # TODO: change this to logging
    logger.log("Average reward for expert rollouts: %f" % np.mean([np.sum(p['rewards']) for p in expert_rollouts]))
    return expert_rollouts


def process_samples_with_reward_extractor(samples, discriminator, batch_size=50000, obs_filter=lambda x: x):
    t0 = time.time()
    super_all_datas = []
    
    for sample in samples:
        sample["observations"] = np.vstack([obs_filter(step) for step in sample["observations"]])
    # splits = []
    # convert all the data to the proper format, concat frames if needed
    super_all_datas = np.vstack([obs_filter(x) for sample in samples for x in sample['observations']])

    extracted_rewards = []
    for batch in batchify_list(super_all_datas, batch_size): #TODO: make batch_size configurable
        extracted_rewards.extend(discriminator.get_reward(batch)) #TODO: unnecessary computation here

    index = 0
    extracted_rewards = np.vstack(extracted_rewards)
    for sample in samples:#len(extracted_rewards):
        sample['true_rewards'] = sample['raw_rewards']
        num_obs = len(sample['observations'])
        sample['raw_rewards'] = select_from_tensor(extracted_rewards, index, index+num_obs).reshape(-1)
        sample['rewards'] = select_from_tensor(extracted_rewards, index, index+num_obs).reshape(-1)
        if len(sample['true_rewards']) != len(sample['rewards']):
            import pdb; pdb.set_trace()
            raise Exception("Problem, extracted rewards not equal in length to old rewards!")
        index += num_obs

    t1 = time.time()
    logger.log("Time to process samples: %f" % (t1-t0))
    return samples
