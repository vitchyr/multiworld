import roboverse as rv

space_mouse = rv.devices.SpaceMouse()
env = rv.make('SawyerSoup-v0', gui=True, gripper_bounds=[0,1])

while True:
	next_obs, rew, term, info = env.step(space_mouse.control, space_mouse.control_gripper)
	print(rew)
	if term: break

pdb.set_trace()
