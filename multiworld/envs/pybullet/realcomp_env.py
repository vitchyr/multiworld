import realcomp
import gym

from collections import OrderedDict
import logging

import numpy as np
from gym import spaces
from pygame import Color
import random
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)

from multiworld.core.wrapper_env import ProxyEnv

import time

class RealCompEnv(ProxyEnv, Serializable):
    def __init__(self, challenge="2D"):
        self.quick_init(locals())
        env = gym.make("REALComp-v0")
        super().__init__(env)
        self.take_goal_image = False
        self.has_set_goal = False
        self.challenge = challenge

    def get_transformed_obs(self, raw_obs):
        # obs['goal']: (240, 320, 3)
        # obs['joint_positions']: list (9)
        # obs['touch_sensors']: (5, )
        # obs['retina']: (240, 320, 3)
        raw_obs['joint_positions'] = np.array(raw_obs['joint_positions'])
        return raw_obs

    def reset(self):
        obs = self.wrapped_env.reset()
        obs = self.get_transformed_obs(obs)
        self.image_observation = obs['retina']
        self.take_goal_image = False
        return obs

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self.get_transformed_obs(obs)
        self.image_observation = obs['retina']
        return new_obs, reward, done, info

    def compute_rewards(self, actions, obs):
        if self.has_set_goal:
            r = self.wrapped_env.evaluateGoal()[1]
        else:
            r = 0
        return np.array([r])

    def get_image(self, width, height):
        if self.take_goal_image:
            return self.image_observation[::5, 40:280:5, :]
        else:
            obs = self.wrapped_env.get_observation()
            return obs["goal"][::5, 40:280:5, :]

    def initialize_camera(self, init_camera):
        pass

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def sample_goals(self, batch_size):
        return dict()

    def set_to_goal(self, goal):
        self.wrapped_env.set_goal()
        self.take_goal_image = True
        self.has_set_goal = True
        return None

    #### Underlying state goal functions

    def set_goal(self, goal):
        pass

    def get_goal(self):
        return None

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass

    # def compute_rewards, get_goal, sample_goals

