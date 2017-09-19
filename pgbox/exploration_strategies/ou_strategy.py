import numpy as np
import numpy.random as nr
from gym.spaces import Box

class OUStrategy(object):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    def __init__(self, env_spec, mu=0, theta=0.15, sigma=0.2, **kwargs):
        assert isinstance(env_spec.action_space, Box)
        assert len(env_spec.action_space.shape) == 1
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = env_spec.action_space
        self.action_space_flat_dim = np.prod(env_spec.action_space.shape)
        self.state = np.ones(self.action_space_flat_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_space_flat_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    def act(self, t, observation, policy, sess, **kwargs):
        action, _ = policy.act(observation, sess)
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, self.action_space.low, self.action_space.high)
