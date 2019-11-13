import time
import numpy as np
import roboverse as rv

num_steps = 100
num_processes = 1

if num_processes == 1:
	env = rv.make('SawyerLift-v0', gui=False, gripper_bounds=[-1,1], action_scale=.2, action_repeat=20, timestep=1./120)
else:
	env = rv.make('ParallelSawyerLift-v0', num_processes=num_processes, gui=False, gripper_bounds=[-1,1], action_scale=.2, action_repeat=20, timestep=1./120)

t0 = time.time()
for _ in range(num_steps):
	act = env.action_space.sample()
	if num_processes > 1:
		act = np.repeat(act[None], num_processes, 0)
	next_obs, rew, term, info = env.step(act)
	# if term: env.reset()
total_time = time.time() - t0
steps_per_sec = num_steps * num_processes / total_time

print('Total: {} | Steps per second: {}'.format(total_time, steps_per_sec))
