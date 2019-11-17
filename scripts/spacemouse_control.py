import roboverse as rv

spacemouse = rv.devices.SpaceMouse()
env = rv.make('SawyerLift2d-v0', gui=True)

while True:
	action = spacemouse.get_action()
	next_obs, rew, term, info = env.step(action)
	print(rew)
	if term: break
