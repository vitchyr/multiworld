import numpy as np
import pdb

import roboverse
import roboverse.devices as devices
import roboverse.bullet as bullet

space_mouse = devices.SpaceMouse()

env = roboverse.make('SawyerLid-v0', gui=True, gripper_bounds=[0,1])

while True:
	next_obs, rew, term, info = env.step(space_mouse.control, space_mouse.control_gripper)
	print(rew)
	if term: break
