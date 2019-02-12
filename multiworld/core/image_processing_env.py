from multiworld.core.wrapper_env import ProxyEnv
import cv2
import warnings
from PIL import Image
from gym.spaces import Box, Dict
import numpy as np

class ImageProcessingEnv(ProxyEnv): # MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
    ):
        """Minimal env to convert a gym env to one with dict observations"""
        self.quick_init(locals())
        super().__init__(wrapped_env)

        obs_box = wrapped_env.observation_space
        image_length = 84 * 84 * 3
        state_length = 0 # self.observation_space.shape[0]
        obs_length = image_length + state_length
        img_space = Box(-10, 10, (obs_length,), dtype=np.uint8)
        self.observation_space = img_space

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        image = self.process(obs)
        return image, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        image = self.process(obs)
        return image

    def process(self, obs):
        return cv2.resize(obs, (84, 110))[26:110, :].transpose().flatten()
