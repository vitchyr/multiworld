import time
import roboverse as rv

num_steps = 100
env = rv.make('SawyerLift-v0', gui=False, gripper_bounds=[0,1])

t0 = time.time()
for _ in range(num_steps):
	act = env.action_space.sample()
	next_obs, rew, term, info = env.step(act)
	if term: env.reset()
total_time = time.time() - t0
steps_per_sec = num_steps / total_time

print('Total: {} | Steps per second: {}'.format(total_time, steps_per_sec))
