import random

import cv2
import numpy as np
import warnings
from PIL import Image
from gym.spaces import Box, Dict

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

from gym.spaces import Box, Dict

class GymToMultiEnv(ProxyEnv): # MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
    ):
        """Minimal env to convert a gym env to one with dict observations"""
        self.quick_init(locals())
        super().__init__(wrapped_env)

        obs_box = wrapped_env.observation_space
        self.observation_space = Dict([
            ('observation', obs_box),
            ('state_observation', obs_box),
        ])

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = dict(
            observation=obs,
            state_observation=obs,
        )
        return new_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        new_obs = dict(
            observation=obs,
            state_observation=obs,
        )
        return new_obs

    def _get_obs(self):
        raise NotImplementedError()
