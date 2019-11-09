import time
import argparse
import numpy as np
import multiprocessing as mp
import gym
import pdb

import roboverse as rv

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='SawyerLift-v0')
parser.add_argument('--savepath', type=str, default='data/dump/')
parser.add_argument('--gui', type=rv.utils.str2bool, default=None)
parser.add_argument('--render', type=rv.utils.str2bool, default=None)
parser.add_argument('--horizon', type=int, default=100)
parser.add_argument('--num_episodes', type=int, default=1)
parser.add_argument('--num_processes', type=int, default=20)
args = parser.parse_args()

# class ParallelEnv:

# 	def __init__(self, env, num_processes=1, **kwargs):
# 		self._env_fn = lambda: gym.make(env, **kwargs)
# 		self._num_processes = num_processes
# 		self._act_queues = [mp.Queue() for _ in range(num_processes)]
# 		self._obs_queues = [mp.Queue() for _ in range(num_processes)]
# 		self._processes = [mp.Process(target=worker, args=(self._env_fn, q1, q2)) 
# 			for q1, q2 in zip(self._act_queues, self._obs_queues)]
# 		[p.start() for p in self._processes]
# 		self._set_spaces()

# 	def _set_spaces(self):
# 		env = self._env_fn()
# 		obs_space = env.observation_space
# 		act_space = env.action_space

# 		obs_low = obs_space.low[None].repeat(self._num_processes, 0)
# 		obs_high = obs_space.high[None].repeat(self._num_processes, 0)

# 		act_low = act_space.low[None].repeat(self._num_processes, 0)
# 		act_high = act_space.high[None].repeat(self._num_processes, 0)

# 		self.observation_space = type(obs_space)(obs_low, obs_high)
# 		self.action_space = type(act_space)(act_low, act_high)
# 		env.close()

# 	def step(self, vec_action):
# 		for act, act_queue in zip(vec_action, self._act_queues):
# 			act_queue.put(act)
# 		outs = [obs_queue.get() for obs_queue in self._obs_queues]

# 		obs = np.stack([out[0] for out in outs])
# 		rew = np.stack([out[1] for out in outs])
# 		term = np.stack([out[2] for out in outs])
# 		info = [out[3] for out in outs]
# 		return obs, rew, term, info



# def worker(env_fn, act_queue, obs_queue):
# 	# seed = int( (time.time() % 1)*1e9 )
# 	# np.random.seed(seed)

# 	env = env_fn()
# 	while True:
# 		action = act_queue.get()
# 		obs, rew, term, info = env.step(action)
# 		obs_queue.put((obs, rew, term, info))


env = gym.make('ParallelSawyerLift-v0', num_processes=args.num_processes, action_scale=.2, action_repeat=20, timestep=1./120, gui=args.gui)
# env = ParallelEnv(args.env, args.num_processes, action_scale=.2, action_repeat=20, timestep=1./120, gui=args.gui)
# parallel_env = env_fn()

t0 = time.time()
for i in range(args.horizon):
	act = env.action_space.sample()
	act = [act for _ in range(args.num_processes)]
	next_obs, rew, term, info = env.step(act)
	print(i, rew.shape)

# obs = env.reset([0,2])
# pdb.set_trace()


# pdb.set_trace()

# policy = rv.policies.GraspingPolicy(env, env._sawyer, env._cube)
# env_fn = rv.utils.Meta(rv.make, args.env, action_scale=.2, action_repeat=20, timestep=1./120, gui=args.gui)

# policy_fn = rv.utils.Meta(rv.policies.GraspingPolicy)

# params = env.get_params()
# timestamp = rv.utils.timestamp()
# pool.save(params, args.savepath, '{}_pool_{}.pkl'.format(timestamp, pool.size))

# parents = []
# children = []
# processes = []

# manager = mp.Manager()
# return_dict = manager.dict()
# for i in range(args.num_processes):
# 	# parent, child = mp.Pipe()
# 	# parents.append(parent)
# 	# children.append(child)
# 	# return_dict = manager.dict()

# 	proc = mp.Process(target=sample, args=(i, env_fn, policy_fn, return_dict))
# 	processes.append(proc)

# t0 = time.time()
# [proc.start() for proc in processes]
# [proc.join() for proc in processes]

total_time = time.time() - t0
total_steps = args.num_processes * args.horizon
steps_per_second = total_steps / total_time

print('total_time: {}'.format(total_time))
print('total_steps: {}'.format(total_steps))
print('steps per second: {}'.format(steps_per_second))
# images = np.concatenate([return_dict[i] for i in range(args.num_processes)], axis=1)
# print(images.shape)
# rv.utils.save_video('dump/parallel_test.avi', images)
pdb.set_trace()
# for proc in 



