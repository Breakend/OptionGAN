# OptionGAN: Learning Joint Reward-Policy Options using Generative Adversarial Inverse Reinforcement Learning

Reinforcement learning has shown promise in learning policies that can solve complex problems. However, manually specifying a good reward function can be difficult, especially for intricate tasks. Inverse reinforcement learning offers a useful paradigm to learn the underlying reward function directly from expert demonstrations. Yet in reality, the corpus of demonstrations may contain trajectories arising from a diverse set of underlying reward functions. Thus, in inverse reinforcement learning, it is useful to consider such a decomposition. The options framework is specifically designed to perform a similar decomposition over the space of underlying policies. We therefore extend the options framework and propose a method to simultaneously recover reward options in addition to policy options. We leverage adversarial methods to simultaneously learn joint reward-policy options using only observed experts states. We show that this approach works well in both simple and complex continuous control tasks and shows significant performance increases in one-shot transfer learning.

## References

A lot of the code here was borrowed from a bunch of sources for the best performing results. As inspiration, we used some of the following repositories.

https://github.com/kvfrans/parallel-trpo

https://github.com/joschu/modular_rl/

https://github.com/openai/baselines

https://github.com/openai/imitation

https://github.com/bstadie/third_person_im

https://github.com/rll/rllab

## Running

To run experiments, use the run scripts in the main directory.

## Notice

We're still in the process of cleaning up the code and merging in changes from private repositories for final publication, so if there's something weird going on let us know.
