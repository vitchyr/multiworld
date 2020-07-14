from roboverse.envs.sawyer_reach import SawyerReachEnv
import roboverse.bullet
import roboverse as rv
import sys
import numpy as np 
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP


#Dictionary mapping keyboard commands to actions
char_to_action = {
    'w': (np.array([0, -1, 0]), 'x'),
    'a': (np.array([1, 0, 0]), 'x'),
    's': (np.array([0, 1, 0]), 'x'),
    'd': (np.array([-1, 0, 0]), 'x'),
    'q': (np.array([1, -1, 0]), 'x'),
    'e': (np.array([-1, -1, 0]), 'x'),
    'z': (np.array([1, 1, 0]), 'x'),
    'c': (np.array([-1, 1, 0]), 'x'),
    'k': (np.array([0, 0, 1]), 'x'),
    'j': (np.array([0, 0, -1]), 'x'),
    'h': (np.array([1, 0, 0, 0]), 'theta'),
    'l': (np.array([-1, 0, 0, 0]), 'theta'),
    'c': (np.array([0, 0, 0, 0]), 'theta'), 
    'u': (0, 'gripper'),
    'i': (1, 'gripper'),
    'r': 'reset'
}

#Dicionary storing whether each key was held
pressed_keys = {
    'w': False,
    'a': False,
    's': False,
    'd': False,
    'q': False,
    'e': False,
    'z': False,
    'c': False,
    'k': False,
    'j': False,
    'h': False,
    'l': False,
    'c': False,
    'p': False,
    'u': False,
    'i': False
}

env = rv.make('SawyerRigMultiobj-v0', gui=True)
#env = SawyerReachEnv(renders=True, control_xyz_position_only=False)
env.reset()
pygame.init()
screen = pygame.display.set_mode((100, 100))
time = pygame.time.get_ticks()
gripper = 0

while True:
    dx = np.array([0, 0, 0])
    dtheta = np.array([0, 0, 0, 0])
    
    #record key presses
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()
        if event.type == KEYDOWN:
            pressed = chr(event.dict['key'])
            if pressed in pressed_keys.keys():
                pressed_keys[pressed] = True
            elif pressed == 'r':
                env.reset()
        if event.type == KEYUP:
            released = chr(event.dict['key'])
            if released in pressed_keys.keys():
                pressed_keys[released] = False
    
    #take actions corresponding to key presses
    for i in pressed_keys.items():
        if i[1]:
            new_action = char_to_action.get(i[0], None)
            if new_action[1] == 'x':
                dx += new_action[0]
            elif new_action[1] == 'theta':
                dtheta += new_action[0]
            elif new_action[1] == 'gripper':
                gripper = new_action[0]
    action = np.concatenate((0.5 * dx, 0.2 * dtheta), axis=0)
    obs, reward, done, info = env.step(action, gripper)
