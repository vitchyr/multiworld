import time
import os
import argparse
import numpy as np
import roboverse as rv
import roboverse.bullet as bullet
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='SawyerSoup-v0')
parser.add_argument('--gui', type=rv.utils.str2bool, default=True)
parser.add_argument('--load_path', type=str, default='data/SawyerSoup2d-v0/scale1-rep10-step1/2019-11-17T17-29-10_states/')
args = parser.parse_args()

env = rv.make(args.env, action_scale=.2, action_repeat=10, timestep=1./120, gui=args.gui)

## 20, 32, 40

for ep in range(1):
	# for i in range(100):
	for i in [20, 32, 40, 40]:
		print(i)
		time.sleep(1)
		fullpath = os.path.join(args.load_path, '{}_{}.bullet'.format(ep, i))
		print(fullpath)
		env.load_state(fullpath)
pdb.set_trace()

