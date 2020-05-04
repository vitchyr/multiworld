import time
import os
import argparse
import numpy as np
import roboverse as rv
import pdb
import gym


env = rv.make("SawyerGraspOne-v0", gui=True)

for i in range(10):
	env.reset()
	for t in range(100):
		env.step()
