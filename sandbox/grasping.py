import numpy as np
import pdb

import roboverse
import roboverse.policies as policies
import roboverse.utils as utils


env = roboverse.make('SawyerLift-v0', render=True)

policy = policies.GraspingPolicy(env, env._sawyer, env._cube)

images = []
for i in range(350):
	policy.control()
	img = env.render()
	utils.save_image('dump/{}.png'.format(i), img)
	images.append(img)
utils.save_video('dump/0.avi', images)