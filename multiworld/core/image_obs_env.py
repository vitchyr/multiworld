from multiworld.core.wrapper_env import ProxyEnv
import cv2
import warnings
from PIL import Image
from gym.spaces import Box, Dict
import numpy as np

class ImageObsEnv(ProxyEnv): # MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
    ):
        """Minimal env to convert a gym env to one with dict observations"""
        self.quick_init(locals())
        super().__init__(wrapped_env)

        obs_box = wrapped_env.observation_space
        image_length = 84 * 84 * 3
        state_length = self.observation_space.shape[0]
        obs_length = image_length + state_length
        img_space = Box(-10, 10, (obs_length,), dtype=np.float32)
        self.observation_space = img_space

    def step(self, action):
        state, reward, done, info = self.wrapped_env.step(action)
        image = self._get_obs()
        obs = np.concatenate((image, state))
        return obs, reward, done, info

    def reset(self):
        state = self.wrapped_env.reset()
        image = self._get_obs()
        obs = np.concatenate((image, state))
        return obs

    def _get_obs(self):
        array = self.wrapped_env.render("rgb_array")
        import pdb; pdb.set_trace()
        return array
