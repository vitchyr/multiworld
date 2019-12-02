from gym.spaces import Box, Dict
import numpy as np

from multiworld.core.wrapper_env import ProxyEnv


class FlatEnv(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            obs_keys=None,
    ):
        self.quick_init(locals())
        super(FlatEnv, self).__init__(wrapped_env)

        self.obs_keys = obs_keys

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        flat_obs = np.hstack([obs[k] for k in self.obs_keys])
        return flat_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        return np.hstack([obs[k] for k in self.obs_keys])
