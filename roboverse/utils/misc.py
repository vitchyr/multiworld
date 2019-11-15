import os
import datetime
import numpy as np
import pickle
from distutils.util import strtobool

def timestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider, dtd=datetime_divider))

def str2bool(x):
    return bool(strtobool(x))

class DemoPool:

    def __init__(self, max_size=1e6):
        self._trajectories = []
        self._max_size = int(max_size)
        self._num_trajectories = 0

    @property
    def size(self):
        return self._num_trajectories

    def save(self, params, *savepath):
        savepath = os.path.join(*savepath)
        print('[ DemoPool ] Saving {0} trajectories'.format(self._num_trajectories))
        pickle.dump(self._trajectories, open(savepath, 'wb+'))

        ## save params
        params_path = savepath.replace('pool', 'params')
        pickle.dump(params, open(params_path, 'wb+'))

    def add_trajectory(self, t):
        if self._num_trajectories != self._max_size:
            self._trajectories.append(t.get_samples())
        self._advance()

    def _advance(self):
        self._num_trajectories = min(self._num_trajectories + 1, self._max_size)

    def get_trajectory(self, i):
        if i >= self._num_trajectories:
            return None
        return self._trajectories[i]

class Trajectory:
    def __init__(self, max_size=1e6):
        self._keys = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
        self._pointer = 0
        self._fields = {}
        for k in self._keys:
            self._fields[k] = []
        self._size = 0
        self._max_size = int(max_size)

    @property
    def size(self):
        return self._size

    def add_sample(self, *arrays):
        if self.size < self._max_size:
            self._add(arrays)
            self._advance()

    def _add(self, arrays):
        for key, array in zip(self._keys, arrays):
            self._fields[key].append(array)

    def _advance(self):
        self._size = min(self._size + 1, self._max_size)
        self._pointer = (self._pointer + 1) % self._max_size

    def get_samples(self):
        return self._fields

class Meta:

    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        self._kwargs.update(**kwargs)
        return self._fn(*args, *self._args, **self._kwargs)
