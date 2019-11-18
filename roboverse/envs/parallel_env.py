import time
import numpy as np
import multiprocessing as mp
import dill
import gym
import pdb

from roboverse.utils import init_from_demos

_RESET = -1
_RENDER = -2
_GET_PARAMS = -3

class _RESET_FROM_STATE:

    def __init__(self, state):
        self._state = state

    def get_state(self):
        return self._state

class ParallelEnv(gym.Env):

    def __init__(self, env=None, num_processes=1, **kwargs):
        self._env_fn = lambda: gym.make(env, **kwargs)
        self._env = self._env_fn()
        self._num_processes = num_processes
        self._act_queues = [mp.Queue() for _ in range(num_processes)]
        self._obs_queues = [mp.Queue() for _ in range(num_processes)]
        self._current_obs = [None for _ in range(self._num_processes)]
        self._processes = [mp.Process(
            target=worker, 
            args=(self._env_fn, self._act_queues[i], self._obs_queues[i], i))
            for i in range(num_processes)] 
        [p.start() for p in self._processes]
        self._set_spaces()

    def _set_spaces(self):
        obs_space = self._env.observation_space
        act_space = self._env.action_space

        self.observation_space = obs_space
        self.action_space = act_space

    def check_params(self, params):
        return self._env.check_params(params)

    def get_params(self):
    	self._act_queues[0].put(_GET_PARAMS)
    	params = self._obs_queues[0].get()
    	return params

    @property
    def num_processes(self):
        return self._num_processes

    @property
    def parallel(self):
    	return True
    
    def step(self, vec_action):
        assert len(vec_action) == self._num_processes

        for act, act_queue in zip(vec_action, self._act_queues):
            act_queue.put(act)
        outs = [obs_queue.get() for obs_queue in self._obs_queues]

        obs = np.stack([out[0] for out in outs])
        rew = np.stack([out[1] for out in outs])
        term = np.stack([out[2] for out in outs])
        info = [out[3] for out in outs]
        self._current_obs = obs.copy()
        return obs, rew, term, info

    # def _compile_infos(self, infos):
    # 	keys = infos[0].keys()
    # 	compiled = {key: [info[key] for info in infos] for key in keys}
    # 	return compiled

    def reset(self, inds):
        for i in inds:
            self._act_queues[i].put(_RESET)
        for i in inds:
            obs = self._obs_queues[i].get()
            self._current_obs[i] = obs
        return self._current_obs

    def reset_from_states(self, inds, states):
        for i, state in zip(inds, states):
            self._act_queues[i].put(_RESET_FROM_STATE(state))
        for i in inds:
            obs = self._obs_queues[i].get()
            self._current_obs[i] = obs
        return self._current_obs

    def set_reset_hook(self, fn):
        fn_serialized = dill.dumps(fn)
        for act_queue in self._act_queues:
            act_queue.put(fn_serialized)
        for obs_queue in self._obs_queues:
            ret = obs_queue.get()
            assert ret

    def render(self, *args, **kwargs):
        for act_queue in self._act_queues:
            act_queue.put(_RENDER)
        images = []
        for obs_queue in self._obs_queues:
            img = obs_queue.get()
            images.append(img)
        images = np.concatenate(images, axis=1)
        return images

def worker(env_fn, act_queue, obs_queue, proc_num):
    seed = int( (time.time() % 1)*1e9 )
    np.random.seed(seed)

    env = env_fn()
    while True:
        action = act_queue.get()
        if type(action) == type(_RESET) and action == _RESET:
            obs = env.reset()
            obs_queue.put(obs)

        elif type(action) == type(_RENDER) and action == _RENDER:
            img = env.render()
            obs_queue.put(img)

        elif type(action) == type(_GET_PARAMS) and action == _GET_PARAMS:
            params = env.get_params()
            obs_queue.put(params)

        elif type(action) == _RESET_FROM_STATE:
            state = action.get_state()
            obs = env.load_state(state)
            obs_queue.put(obs)

        elif type(action) == bytes:
            fn = dill.loads(action)
            env.set_reset_hook(fn)
            obs_queue.put(True)

        else:
            obs, rew, term, info = env.step(action)
            obs_queue.put((obs, rew, term, info))


