import numpy as np
import pdb

import roboverse
import roboverse.devices as devices

space_mouse = devices.SpaceMouse()

env = roboverse.make('SawyerLift-v0', render=True, gripper_bounds=[0,1])

while True:
	env.step(space_mouse.control, space_mouse.control_gripper)