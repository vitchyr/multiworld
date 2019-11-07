import time
import argparse
import numpy as np
import multiprocessing as mp
import roboverse as rv
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='SawyerLift-v0')
parser.add_argument('--savepath', type=str, default='data/dump/')
parser.add_argument('--gui', type=rv.utils.str2bool, default=None)
parser.add_argument('--render', type=rv.utils.str2bool, default=None)
parser.add_argument('--horizon', type=int, default=500)
parser.add_argument('--num_episodes', type=int, default=1)
parser.add_argument('--num_processes', type=int, default=10)
args = parser.parse_args()

def sample(proc_num, env_fn, policy_fn, return_dict):
	seed = int( (time.time() % 1)*1e9 )
	np.random.seed(seed)
	# print(np.random.randn(5), t)

	env = env_fn()
	policy = policy_fn(env, env._sawyer, env._cube)

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
				print('min_grasp_step: ', min_grasp_step)
			next_obs, rew, term, info = env.step(act)
			pool.add_sample(obs, act, next_obs, rew, term)
			obs = next_obs

			print(i, rew, term)
			ep_rew += rew
			if args.render:
				img = env.render()
				images.append(img)
				# rv.utils.save_image('{}/{}.png'.format(args.savepath, i), img)

			if term: break

		print('Episode: {} | steps: {} | return: {} | min grasp step: {}'.format(ep, i+1, ep_rew, min_grasp_step))
		if args.render:
			rv.utils.save_video('{}/{}.avi'.format(args.savepath, ep), images)

	# pool._prune()
	# obs = pool._fields['observations']
	samples = pool.get_samples()
	return_dict[proc_num] = samples

env = rv.make(args.env, action_scale=.2, action_repeat=20, timestep=1./120, gui=args.gui,)
env_fn = env.get_constructor()

# pdb.set_trace()

# policy = rv.policies.GraspingPolicy(env, env._sawyer, env._cube)
# env_fn = rv.utils.Meta(rv.make, args.env, action_scale=.2, action_repeat=20, timestep=1./120, gui=args.gui)

policy_fn = rv.utils.Meta(rv.policies.GraspingPolicy)

# params = env.get_params()
# timestamp = rv.utils.timestamp()
# pool.save(params, args.savepath, '{}_pool_{}.pkl'.format(timestamp, pool.size))

parents = []
children = []
processes = []

manager = mp.Manager()
return_dict = manager.dict()
for i in range(args.num_processes):
	# parent, child = mp.Pipe()
	# parents.append(parent)
	# children.append(child)
	# return_dict = manager.dict()

	proc = mp.Process(target=sample, args=(i, env_fn, policy_fn, return_dict))
	processes.append(proc)

t0 = time.time()
[proc.start() for proc in processes]
[proc.join() for proc in processes]

total_time = time.time() - t0
total_steps = sum([len(return_dict[i]['observations']) for i in range(args.num_processes)])
steps_per_second = total_steps / total_time

print('total_time: {}'.format(total_time))
print('total_steps: {}'.format(total_steps))
print('steps per second: {}'.format(steps_per_second))
# images = np.concatenate([return_dict[i] for i in range(args.num_processes)], axis=1)
# print(images.shape)
# rv.utils.save_video('dump/parallel_test.avi', images)
pdb.set_trace()
# for proc in 



