import roboverse as rv

spacemouse = rv.devices.SpaceMouse()

env = rv.make('SawyerSoup2d-v0', action_repeat=4, gui=True)
print(env.observation_space, env.action_space)

while True:
	action = spacemouse.get_action()
	next_obs, rew, term, info = env.step(action)
	print(rew, next_obs.shape)
	print(next_obs)
	if term: break

pdb.set_trace()
