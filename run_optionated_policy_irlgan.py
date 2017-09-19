import numpy as np
import tensorflow as tf
import gym
import gym_extensions
from gym_extensions.wrappers.normalized_env import normalize
from pgbox.utils import *
from pgbox.trpo.model import *
import argparse
from pgbox.trpo.rollouts import *
import json
from pgbox.parallel_algo_utils import *
from pgbox.policies.gated_gaussian_mlp_policy import *
from irlbox.utils import load_expert_rollouts
from irlbox.irlgan.trainer import ParallelTrainer
from irlbox.discriminators.mlp_gated_discriminator import OptionatedMLPDiscriminator
import roboschool
from pgbox.valuefunctions.nn_vf import *
from pgbox.sampling_utils import rollout
import pickle


parser = argparse.ArgumentParser(description='TRPO.')
# these parameters should stay the same
parser.add_argument("task", type=str)
parser.add_argument("expert_rollouts_path", type=str)
parser.add_argument("--num_expert_rollouts", type=int, default=10)
parser.add_argument("--timesteps_per_batch", type=int, default=25000)
parser.add_argument("--n_iters", type=int, default=750)
parser.add_argument("--gamma", type=float, default=.995)
parser.add_argument("--max_kl", type=float, default=.01)
parser.add_argument("--cg_damping", type=float, default=1e-1)
parser.add_argument("--num_threads", type=int, default=2)
parser.add_argument("--monitor", type=bool, default=False)
parser.add_argument("--lam", type=float, default=.97)
parser.add_argument("--use_reward_filter", action="store_true", help="Turn of default of original TRPO code.")
parser.add_argument("--use_obs_filter", action="store_true", help="Turn of default of original TRPO code.")
parser.add_argument("--discriminator_lr", default=1e-3, type=float)
parser.add_argument("--log_dir", default="./logs/")
parser.add_argument("--num_options", default=2, type=int)
parser.add_argument("--d_cross_entropy_reweighting", default=1.0, type=float)
parser.add_argument("--d_gradient_penalty_weight", default=0.0, type=float)
parser.add_argument("--d_l2_penalty_weight", default=0.0, type=float)
parser.add_argument("--d_mutual_info_penalty_weight", default=0.1, type=float)
parser.add_argument("--d_num_epochs_per_step", default=1, type=int)
parser.add_argument("--d_lambda_s", default=10.0, type=float)
parser.add_argument("--d_lambda_v", default=1.0, type=float)
parser.add_argument("--d_gate_change_penalty", default=0.0, type=float)
parser.add_argument("--discriminator_size", nargs="+", default=(64,64), type=int)
parser.add_argument("--policy_size", nargs="+", default=(64,64), type=int)
parser.add_argument("--gate_size", nargs="+", default=(32,32), type=int)
parser.add_argument("--concat_prev_timestep", action="store_true")
parser.add_argument("--d_ent_reg_weight", default=0.0, type=float)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--use_ppo", action="store_true")

activation_map = { "relu" : tf.nn.relu, "selu" : selu,  "leaky_relu" : leaky_relu, "tanh" :tf.nn.tanh, "prelu" : prelu}
args = parser.parse_args()

print("All Args")
print(args)

args.activation = activation_map[args.activation]

logger.add_text_output(args.log_dir + "debug.log")
logger.add_tabular_output(args.log_dir + "progress.csv")

learner_env = gym.make(args.task)

expert_rollouts = load_expert_rollouts(args.expert_rollouts_path, max_traj_len = -1, num_expert_rollouts = args.num_expert_rollouts)

if args.concat_prev_timestep:
    from pgbox.sampling_utils import apply_transformers
    transformers = [
        gym_extensions.wrappers.transformers.AppendPrevTimeStepTransformer()]
    learner_env = gym_extensions.wrappers.observation_transform_wrapper.ObservationTransformWrapper(
        learner_env, transformers)
    apply_transformers(expert_rollouts, transformers)
    args.transformers = transformers
else:
    args.transformers = None

policy = GatedGaussianMLPPolicy(learner_env, hidden_sizes=args.policy_size, activation=args.activation, gate_hidden_sizes=args.gate_size, num_options=args.num_options)

baseline = MLPValueFunction(learner_env, hidden_sizes=args.policy_size, activation=tf.nn.tanh)

if args.use_ppo:
    trpo = ParallelPPO(learner_env, policy, args, vf=baseline)
else:
    trpo = ParallelTRPO(learner_env, policy, args, vf=baseline)

discriminator = OptionatedMLPDiscriminator(learner_env.observation_space.shape[0],
                                           hidden_sizes=args.discriminator_size,
                                           activation=args.activation,
                                           learning_rate=args.discriminator_lr,
                                           num_options=args.num_options,
                                           cross_entropy_reweighting=args.d_cross_entropy_reweighting,
                                           gradient_penalty_weight=args.d_gradient_penalty_weight,
                                           l2_penalty_weight = args.d_l2_penalty_weight,
                                           mutual_info_penalty_weight = args.d_mutual_info_penalty_weight,
                                           num_epochs_per_step=args.d_num_epochs_per_step,
                                           lambda_s = args.d_lambda_s,
                                           lambda_v = args.d_lambda_v,
                                           gate_hidden_sizes=args.gate_size,
                                           gate_change_penalty=args.d_gate_change_penalty,
                                           ent_reg_weight = args.d_ent_reg_weight
                                           )

trainer = ParallelTrainer(trpo, expert_rollouts, discriminator, policy, args)

iterations = 0
while iterations <= args.n_iters:
    iterations = trainer.step()
trainer.end()

if True:
    with tf.Session() as session:
        sample_id = 0
        sample_ids = []
        learner_env.reset()
        policy.set_param_values(session, trpo.policy_weights)
        path = rollout(learner_env, policy, 2000, session, collect_images=True)
        images = path.pop('images')
        for image, obs in zip(images, path["observations"]):
            scipy.misc.imsave('./images/sample_%d.png' % sample_id, image)
            sample_ids.append(sample_id)
            sample_id += 1
        path["sampled_ids"] = sample_ids
        with open("data.pickle", "wb") as output_file:
            pickle.dump(path, output_file)

        print("Path activations:")
        print(path["info"]["gate_dist"])

        ave_gating_activations_per_rollout = []
        for path in expert_rollouts:
            gating_activations = []
            for step in path["observations"]:
                act, info = policy.act(step, session)
                gating_activations.append(info["gate_dist"])
            ave_gating_activations_per_rollout.append(np.mean(np.vstack(gating_activations), axis=0))

        print("gating activations per rollout")
        for i, x in enumerate(ave_gating_activations_per_rollout):
            print(i)
            print(x)
        with open("gating_activations_per_rollout.pickle", "wb") as output_file:
            pickle.dump(ave_gating_activations_per_rollout, output_file)
