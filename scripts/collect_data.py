import argparse
import numpy as np
import roboverse as rv
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='SawyerLift-v0')
parser.add_argument('--savepath', type=str, default='data/grasping/')
parser.add_argument('--gui', type=rv.utils.str2bool, default=None)
parser.add_argument('--render', type=rv.utils.str2bool, default=None)
parser.add_argument('--horizon', type=int, default=500)
parser.add_argument('--num_episodes', type=int, default=2)
args = parser.parse_args()

rv.utils.make_dir(args.savepath)

env = rv.make(args.env, gui=args.gui, gripper_bounds=[0,1])
policy = rv.policies.GraspingPolicy(env, env._sawyer, env._cube)
pool = rv.utils.DemoPool()

for ep in range(args.num_episodes):
	obs = env.reset()
	images = []
	for i in range(args.horizon):
		act = policy.get_action(obs)
		next_obs, rew, term, info = env.step(act)
		pool.add_sample(obs, act, next_obs, rew, term)
		obs = next_obs

		print(rew, term)
		if args.render:
			img = env.render()
			images.append(img)
			rv.utils.save_image('{}/{}.png'.format(args.savepath, i), img)

	if args.render:
		rv.utils.save_video('{}/{}.avi'.format(args.savepath, ep), images)

pool.save(args.savepath, '{}_pool.pkl'.format(rv.utils.timestamp()))