import roboverse
import numpy as np
import time
import roboverse.utils as utils
import argparse

from roboverse.envs.sawyer_grasp_randomized import SawyerGraspOneEnv
from roboverse.envs.sawyer_grasp_v4 import SawyerGraspV4Env

parser = argparse.ArgumentParser()
parser.add_argument("--save_video", action="store_true")
args = parser.parse_args()
# env = SawyerGraspOneEnv(
#     max_force=100,
#     action_scale=0.05,
#     gui=False,
#     randomize=True,
#     # invisible_robot=True,
# )

env = SawyerGraspV4Env(
    max_force=100,
   action_scale=0.05,
   pos_init=[0.7, 0.2, -0.2],
   pos_low=[.5, -.05, -.38],
   pos_high=[.9, .30, -.15],
   object_position_low=(.65, .10, -.20),
   object_position_high=(.80, .25, -.20),
   num_objects=1,
   # height_threshold=-0.3,
   object_ids=[1]
)

    # roboverse.make('SawyerGraspOne-v0', gui=False, randomize=True)
obj_key = 'lego'
num_grasps = 0

env.reset()
# target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
images = []


episode_reward = 0.

for i in range(50):
    ee_pos = env.get_end_effector_pos()
    object_pos = env.get_object_midpoint(obj_key)
    print(object_pos)
    xyz_diff = object_pos - ee_pos
    xy_diff = xyz_diff[:2]
    if np.linalg.norm(xyz_diff) > 0.02:
        action = object_pos - ee_pos
        action *= 5.0
        if np.linalg.norm(xy_diff) > 0.05:
            action[2] *= 0.5
        grip=0.
        print('Approaching')
    elif o[3] > 0.03:
        # o[3] is gripper tip distance
        action = np.zeros((3,))
        grip=0.8
        print('Grasping')
    elif info['object_goal_distance'] > 0.01:
        action = env._goal_pos - object_pos
        print(env._goal_pos)
        action *= 5.0
        grip=1.
        print('Moving')
    else:
        action = np.zeros((3,))
        grip=1.
        print('Holding')

    action = np.append(action, [grip])

    if args.save_video:
        img = env.render()
        images.append(img)

    # time.sleep(0.05)
    o, r, d, info = env.step(action)
    env.render_obs()
    print(action)
    # print(o)
    print(r)
    print('object to goal: {}'.format(info['object_goal_distance']))
    print('object to gripper: {}'.format(info['object_gripper_distance']))
    episode_reward += r

print('Episode reward: {}'.format(episode_reward))
object_pos = env.get_object_midpoint(obj_key)
if object_pos[2] > -0.1:
    num_grasps += 1

if args.save_video:
    utils.save_video('data/lego_test_{}.avi'.format(0), images)
