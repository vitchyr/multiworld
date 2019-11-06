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
		self._keys = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
		self._fields = {}
		self._max_size = int(max_size)
		self._size = 0
		self._pointer = 0

	@property
	def size(self):
		return self._size
	
	def add_field(self, key, val):
		assert key not in self._keys
		self._keys = tuple(list(self._keys) + [key])
		self._fields[key] = val

	def add_sample(self, *arrays):
		if self._size:
			self._add(arrays)
		else:
			self._init(arrays)

		self._advance()
		# print(self._size, self._pointer)

	def save(self, *savepath):
		savepath = os.path.join(*savepath)
		self._prune()
		save_info = [(key, self._fields[key].shape) for key, array in self._fields.items()]
		print('[ DemoPool ] Saving to: {} | {}'.format(savepath, save_info))
		pickle.dump(self._fields, open(savepath, 'wb'))

	def _add(self, arrays):
		for key, array in zip(self._keys, arrays):
			self._fields[key][self._pointer] = array

	def _init(self, arrays):
		for key, array in zip(self._keys, arrays):
			shape = array.shape if type(array) == np.ndarray else (1,)
			dtype = array.dtype if type(array) == np.ndarray else type(array)
			self._fields[key] = np.zeros((self._max_size, *shape), dtype=dtype)
			self._fields[key][self._pointer] = array
			# print(key, self._fields[key].shape, self._fields[key].dtype)

	def _advance(self):
		self._size = min(self._size + 1, self._max_size)
		self._pointer = (self._pointer + 1) % self._max_size

	def _prune(self):
		for key in self._keys:
			self._fields[key] = self._fields[key][:self._size]




