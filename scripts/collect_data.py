import argparse
import numpy as np
import roboverse as rv
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='SawyerLift-v0')
parser.add_argument('--savepath', type=str, default='data/field-test/')
parser.add_argument('--gui', type=rv.utils.str2bool, default=None)
parser.add_argument('--render', type=rv.utils.str2bool, default=None)
parser.add_argument('--horizon', type=int, default=500)
parser.add_argument('--num_episodes', type=int, default=100)
args = parser.parse_args()

rv.utils.make_dir(args.savepath)

env = rv.make(args.env, gui=args.gui)
policy = rv.policies.GraspingPolicy(env, env._sawyer, env._cube)
pool = rv.utils.DemoPool()
print('Observation space: {} | Action space: {}'.format(env.observation_space, env.action_space))

for ep in range(args.num_episodes):
	obs = env.reset()
	ep_rew = 0
	min_grasp_step = None
	images = []
	for i in range(args.horizon):
		act = policy.get_action(obs)
		if act[-1] > 0 and min_grasp_step is None:
			min_grasp_step = i
			print(min_grasp_step)
		next_obs, rew, term, info = env.step(act)
		pool.add_sample(obs, act, next_obs, rew, term)
		obs = next_obs

		# print(rew, term)
		ep_rew += rew
		if args.render:
			img = env.render()
			images.append(img)
			rv.utils.save_image('{}/{}.png'.format(args.savepath, i), img)

		if term: break

	print('Episode: {} | steps: {} | return: {} | min grasp step: {}'.format(ep, i+1, ep_rew, min_grasp_step))
	if args.render:
		rv.utils.save_video('{}/{}.avi'.format(args.savepath, ep), images)

params = env.get_params()
timestamp = rv.utils.timestamp()
pool.save(params, args.savepath, '{}_pool_{}.pkl'.format(timestamp, pool.size))

