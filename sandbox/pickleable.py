import pickle
import pdb
import roboverse as rv
from roboverse import bullet

env = rv.make('SawyerBase-v0')

for x in dir(env):
	attr = getattr(env, x)
	print(type(attr), x)

sawyer = 0
name = bullet.get_joint_info(sawyer, 20, 'joint_name')
print(name)
pdb.set_trace()

pickle.dump(env, open('dump/blah.pkl', 'wb'))