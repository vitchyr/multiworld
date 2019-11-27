import roboverse
import numpy as np
import time
import roboverse.utils as utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_video", action="store_true")
args = parser.parse_args()

env = roboverse.make('SawyerGraspOne-v0', gui=True)
obj_key = 'lego'
num_grasps = 0

env.reset()
# target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
images = []
target_pos = env.get_object_midpoint(obj_key)

target_pos[0] += -0.02
target_pos[1] += 0.00
target_pos[2] += -0.03

for i in range(50):
    ee_pos = env.get_end_effector_pos()

    if i < 25:
        action = target_pos - ee_pos
        action[2] = 0.
        action *= 3.0
        grip=0.
    elif i < 35:
        action = target_pos - ee_pos
        action[2] -= 0.03
        action *= 3.0
        action[2] *= 2.0
        grip=0.
    elif i < 42:
        action = np.zeros((3,))
        grip=0.5
    else:
        action = np.zeros((3,))
        action[2] = 1.0
        grip=1.

    action = np.append(action, [grip])

    if args.save_video:
        img = env.render()
        images.append(img)

    time.sleep(0.05)
    o, r, d, info = env.step(action)
    # print(action)
    # print(o)

object_pos = env.get_object_midpoint(obj_key)
if object_pos[2] > -0.1:
    num_grasps += 1

if args.save_video:
    utils.save_video('data/lego_test_{}.avi'.format(0), images)
