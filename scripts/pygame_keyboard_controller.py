"""
Use this script to control the pygame env with the keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import gym
import sys

import numpy as np
import pygame
from multiworld.envs.pygame.pick_and_place import PickAndPlaceEnv
from pygame.locals import QUIT, KEYDOWN

from multiworld import register_pygame_envs

pygame.init()


char_to_action = {
    'w': np.array([0, -1]),
    'a': np.array([-1, 0]),
    's': np.array([0, 1]),
    'd': np.array([1, 0]),
    'q': np.array([-1, -1]),
    'e': np.array([1, -1]),
    'z': np.array([-1, 1]),
    'c': np.array([1, 1]),
    'x': 'toggle',
    'r': 'reset',
    'j': 'drop',
    'k': 'pickup',
}

register_pygame_envs()
# env = gym.make('Point2D-Big-UWall-v1')
# env = gym.make('FiveObjectPickAndPlaceRandomInit1DEnv-v1')
# env = gym.make('OneObject-PickAndPlace-OnRandomObjectInit-2D-v1')
env = gym.make('OneObject-PickAndPlace-OriginInit-2D-v1')
# env = PickAndPlaceEnv(
#     num_objects=3,
#     render_size=240,
#     render_onscreen=True,
#     # render_onscreen=False,
#     # get_image_base_render_size=(48, 48),
# )
env.render_onscreen=True
env.render_dt_msec = 100
NDIM = env.action_space.low.size
lock_action = False
obs = env.reset()
action = np.zeros(3)
while True:
    done = False
    if not lock_action:
        action[:2] = 0
    for event in pygame.event.get():
        event_happened = True
        if event.type == QUIT:
            sys.exit()
        if event.type == KEYDOWN:
            char = event.dict['key']
            new_action = char_to_action.get(chr(char), None)
            if new_action == 'toggle':
                lock_action = not lock_action
            elif new_action == 'reset':
                done = True
            elif new_action == 'pickup':
                action[2] = 1
            elif new_action == 'drop':
                action[2] = -1
            elif new_action is not None:
                action[:2] = new_action[:2]
            else:
                action = np.zeros(3)
    env.step(action[:NDIM])
    if done:
        obs = env.reset()
    env.render(mode='interactive')
    # img = env.get_image(48, 48)
    # import cv2
    # print(img.shape)
    # cv2.imshow('tmp', img)
    # cv2.waitKey(1)
