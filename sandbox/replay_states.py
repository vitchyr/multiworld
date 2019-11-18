import os
import argparse
import numpy as np
import roboverse as rv
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='SawyerSoup-v0')
parser.add_argument('--gui', type=rv.utils.str2bool, default=True)
parser.add_argument('--load_path', type=str, default='data/SawyerSoup2d-v0/dump/2019-11-17T17-03-42_states')
args = parser.parse_args()

env = rv.make(args.env, action_scale=.2, action_repeat=10, timestep=1./120, gui=args.gui)

for ep in range(1):
	for i in range(100):
		fullpath = os.path.join(args.load_path, '{}_{}.bullet'.format(ep, i))
		env.load_state(fullpath)

