import os
import argparse
import numpy as np
import roboverse as rv
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='ParallelSawyerSoup2d-v0')
parser.add_argument('--loadpath', type=str, default='scale1-rep10-step1/')
parser.add_argument('--gui', type=rv.utils.str2bool, default=None)
parser.add_argument('--render', type=rv.utils.str2bool, default=None)
parser.add_argument('--save_state', type=rv.utils.str2bool, default=None)
parser.add_argument('--horizon', type=int, default=1000)
parser.add_argument('--num_episodes', type=int, default=10)
args = parser.parse_args()

args.loadpath = os.path.join('data', args.env.replace('Parallel', ''), args.loadpath)

timestamp = rv.utils.timestamp()
print('timestamp: {}'.format(timestamp))

reset_states = [
	'data/SawyerSoup2d-v0/scale1-rep10-step1/2019-11-17T17-29-10_states/0_0.bullet',
	'data/SawyerSoup2d-v0/scale1-rep10-step1/2019-11-17T17-29-10_states/0_20.bullet',
	'data/SawyerSoup2d-v0/scale1-rep10-step1/2019-11-17T17-29-10_states/0_32.bullet',
	'data/SawyerSoup2d-v0/scale1-rep10-step1/2019-11-17T17-29-10_states/0_40.bullet',
]
num_processes = len(reset_states)

env = rv.make(args.env, num_processes=num_processes, action_scale=.2, action_repeat=10, timestep=1./120, gui=args.gui)
# reset_fn = rv.utils.init_from_demos(args.loadpath)
# env.set_reset_hook(reset_fn)



spacemouse = rv.devices.SpaceMouse()
pool = rv.utils.DemoPool()
print('Observation space: {} | Action space: {}'.format(env.observation_space, env.action_space))

for ep in range(args.num_episodes):
	obs = env.reset_from_states(range(num_processes), reset_states)
	ep_rew = 0
	images = []
	for i in range(args.horizon):
		act = spacemouse.get_action()
		act = [act for _ in range(num_processes)]
		next_obs, rew, term, info = env.step(act)
		# pool.add_sample(obs, act, next_obs, rew, term)
		
		if args.render:
			img = env.render()
			images.append(img)
			print(i, img.shape)
			rv.utils.save_image('/Users/janner/Desktop/parallel_dump/{}.png'.format(i), img)

		obs = next_obs
		ep_rew += rew

		# print(act, rew, term)
		if term.all(): break
		
	print(ep, i+1, ep_rew)
