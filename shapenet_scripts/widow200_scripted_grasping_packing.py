import roboverse
import numpy as np
import time
import math
import roboverse.utils as utils
import roboverse.bullet as bullet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_video", action="store_true")
args = parser.parse_args()

env = roboverse.make('WidowBoxPackingOne-v0', gui=True)
obj_key = 'lego'
num_grasps = 0

o = env.reset()
# target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
images = []

print(env.get_end_effector_pos())

episode_reward = 0.
holding = False
rotate_object = False
rotate_bowl = False
opened = False

lid_joint_id = bullet.get_index_by_attribute(env._objects['box'], 'joint_name', 'lid_joint')
print("lid_joint_id", lid_joint_id)
lid_joint_pos = np.array(bullet.get_link_state(env._objects['box'], lid_joint_id, 'world_link_pos'))
print("lid_joint_pos", lid_joint_pos)
handle_idx = 1 # start with going for right handle.
ts = 0 # Persistent across all for-loops
reward_per_ts = [] # store rewards across all for-loops

for i in range(1000):
    ee_pos = np.array(env.get_end_effector_pos())
    lego_pos = np.array(env.get_object_midpoint(obj_key))
    box_pos = np.array(env.get_object_midpoint('box'))
    # handle_l_id = bullet.get_index_by_attribute(env._objects['box'], 'link_name', 'handle_l')
    # handle_l_pos = np.array(bullet.get_link_state(env._objects['box'], handle_l_id, 'pos'))
    handle_r_id = bullet.get_index_by_attribute(env._objects['box'], 'link_name', 'handle_r')
    handle_r_pos = np.array(bullet.get_link_state(env._objects['box'], handle_r_id, 'pos'))

    # print("handle_l_pos", handle_l_pos)
    print("box_pos", box_pos)

    # Door angle calc
    _, x, y = (handle_r_pos - lid_joint_pos) # y -> x, z -> y
    door_angle = np.arctan2(y, -x) # equivalent to env.get_door_angle()
    opened = abs(door_angle - (np.pi / 2)) < 0.2 # 0.375
    if opened:
        break
    lost_handle_grip = o[3] < 0.05 and np.linalg.norm(xyz_diff) > 0.07 and door_angle > 0.1
    handle_pos = handle_r_pos
    object_pos = handle_pos if not opened else lego_pos
    print("lost_handle_grip", lost_handle_grip)
    xyz_diff = object_pos - ee_pos
    xy_diff = xyz_diff[:2]
    xy_goal_diff = (env._goal_pos - object_pos)[:2]
    base_angle_diff = abs(utils.angle(object_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0])))
    print("base_angle_diff", base_angle_diff)

    if utils.true_angle_diff(base_angle_diff) > 0.1 \
            and not holding and not rotate_object:
        a = utils.angle(ee_pos[:2], np.array([0.7, 0]))
        diff = utils.true_angle_diff(utils.angle(object_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0])))
        # diff = diff - 2 * math.pi if diff > 2 * math.pi else diff + 2 * math.pi if diff < 0 else diff
        print(diff, 'asdfasfda')
        if diff > math.pi:
            action = np.array([math.sin(a), -math.cos(a), 0.1])
        else:
            action = np.array([math.sin(a), -math.cos(a), 0.1])
        action /= 2.0

        grip = 0
        print('Rotating')
    elif not opened and grip:
        # need some maths here.
        # Calculate angle.
        action = np.array([0, 0.4*y, -0.4*x])
        if lost_handle_grip:# lost grip on object:
            print("Lost grip on object")
            # make it approach and pick up object again
            action = np.array([0, -1, 0])
            grip = 0
        print("door_angle", door_angle)
        print("opening")
    elif np.linalg.norm(xyz_diff) > 0.05 and not holding:
        action = object_pos - ee_pos
        action /= np.linalg.norm(object_pos - ee_pos)
        action /= 3
        grip=0.
        rotate_object = True
        print('Approaching')
    elif o[3] > 0.05 and not holding:
        # o[3] is gripper tip distance
        action = np.zeros((3,))
        action[2] = -0.01
        grip=1.0
        print('Grasping')
    elif env._goal_pos[2] - object_pos[2] > 0.01 and not holding:
        action = env._goal_pos - object_pos
        grip = 1.0
        action[0] = 0
        action[1] = 0
        action *= 3.0
        print('Lifting')
    elif abs(utils.angle(box_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0]))) > 0.1\
            and not holding and not rotate_bowl:
        a = utils.angle(ee_pos[:2], np.array([0.7, 0]))
        diff = utils.angle(box_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0]))
        diff = diff - 2 * math.pi if diff > 2 * math.pi else diff + 2 * math.pi if diff < 0 else diff
        if diff > math.pi:
            action = np.array([math.sin(a), -math.cos(a), 0])
        else:
            action = -np.array([math.sin(a), -math.cos(a), 0])
        action[2] = 0.02
        action /= 2.0
        grip = 1.0
        print('Rotating')
    elif np.linalg.norm((box_pos - object_pos)[:2]) > 0.03:
        action = np.array([1, 1, 1]) * (box_pos - object_pos)
        grip = 1.0
        action *= 3.0
        holding = True
        rotate_bowl = True
        print("action", action)
        print("Moving to Bowl")
        print("np.linalg.norm((box_pos - object_pos)[:2])", np.linalg.norm((box_pos - object_pos)[:2]))
    else:
        action = np.zeros((3,))
        grip=0.
        holding = True
        print('Dropping')



    action = np.append(action, [grip])

    if args.save_video:
        img = env.render()
        images.append(img)

    time.sleep(0.05)
    o, r, d, info = env.step(action)
    # import ipdb; ipdb.set_trace()
    print("info", info)
    print("action", action)
    print(o[3])
    print("reward", r)
    print('object to goal: {}'.format(info['object_goal_distance']))
    print('object to gripper: {}'.format(info['object_gripper_distance']))
    episode_reward += r
    ts += 1
    reward_per_ts.append(r)

# episode_reward = 0.
holding = False
rotate_object = False
# rotate_bowl = False
# opened = False

# for i in range(1000):
#     ee_pos = np.array(env.get_end_effector_pos())
#     lego_pos = np.array(env.get_object_midpoint(obj_key))
#     box_pos = np.array(env.get_object_midpoint('box'))
#     # handle_l_id = bullet.get_index_by_attribute(env._objects['box'], 'link_name', 'handle_l')
#     # handle_l_pos = np.array(bullet.get_link_state(env._objects['box'], handle_l_id, 'pos'))
#     handle_r_id = bullet.get_index_by_attribute(env._objects['box'], 'link_name', 'handle_r')
#     handle_r_pos = np.array(bullet.get_link_state(env._objects['box'], handle_r_id, 'pos'))

#     # print("handle_l_pos", handle_l_pos)
#     print("box_pos", box_pos)

#     # Door angle calc
#     door_angle = env.get_door_angle()
#     opened = abs(door_angle - (np.pi / 2)) < 0.11
#     lost_handle_grip = False # it will never lose grip of the lego
#     handle_pos = handle_r_pos
#     object_pos = lego_pos
#     print("lost_handle_grip", lost_handle_grip)
#     xyz_diff = object_pos - ee_pos
#     xy_diff = xyz_diff[:2]
#     xy_goal_diff = (env._goal_pos - object_pos)[:2]
#     base_angle_diff = abs(utils.angle(object_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0])))
#     print("base_angle_diff", base_angle_diff)
#     lego_dropped = np.linalg.norm(box_pos - object_pos) < 0.03
#     if lego_dropped:
#         break
#     print("utils.true_angle_diff(base_angle_diff)", utils.true_angle_diff(base_angle_diff))
#     if utils.true_angle_diff(base_angle_diff) > 0.1 \
#             and not holding and not rotate_object:
#         a = utils.angle(ee_pos[:2], np.array([0.7, 0]))
#         diff = utils.true_angle_diff(utils.angle(object_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0])))
#         # diff = diff - 2 * math.pi if diff > 2 * math.pi else diff + 2 * math.pi if diff < 0 else diff
#         print(diff, 'asdfasfda')
#         if diff > math.pi:
#             action = -np.array([math.sin(a), -math.cos(a), 0])
#         else:
#             action = np.array([math.sin(a), -math.cos(a), 0])
#             # action = -np.array([math.sin(a), -math.cos(a), 0]) # used to be negative. Now they rotate ccw.
#         action /= 2.0

#         grip = 0
#         print('Rotating')
#     elif np.linalg.norm(xyz_diff) > 0.05 and not holding:
#         action = object_pos - ee_pos
#         action /= np.linalg.norm(object_pos - ee_pos)
#         action /= 3
#         grip=0.
#         rotate_object = True
#         print('Approaching')
#     elif o[3] > 0.05 and not holding:
#         # o[3] is gripper tip distance
#         action = np.zeros((3,))
#         action[2] = -0.01
#         grip=1.0
#         print('Grasping')
#     elif abs(env._goal_pos[2] - object_pos[2]) > 0.01 and not holding:
#         # Something wrong with this condition.
#         # Lifts object to some goal height.
#         print("env._goal_pos[2] - object_pos[2]", env._goal_pos[2] - object_pos[2])
#         print("env._goal_pos[2]", env._goal_pos[2])
#         action = env._goal_pos - object_pos
#         grip = 1.0
#         action[0] = 0
#         action[1] = 0
#         action *= 3.0
#         print('Lifting')
#     elif utils.true_angle_diff(
#             abs(utils.angle(box_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0])))) > 0.05 \
#             and not holding and not rotate_bowl:
#         print("env._goal_pos[2] - object_pos[2]", env._goal_pos[2] - object_pos[2])
#         a = utils.angle(ee_pos[:2], np.array([0.7, 0]))
#         diff = utils.angle(box_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0]))
#         print("diff", utils.true_angle_diff(abs(diff)))
#         diff = diff - 2 * math.pi if diff > 2 * math.pi else diff + 2 * math.pi if diff < 0 else diff
#         if diff > math.pi:
#             action = np.array([math.sin(a), -math.cos(a), 0])
#         else:
#             action = -np.array([math.sin(a), -math.cos(a), 0])
#         action[2] = 0.02
#         action /= 10.0 # 2.0
#         grip = 1.0
#         holding = True
#         print('Rotating1')
#     elif np.linalg.norm((box_pos - object_pos)[:2]) > 0.03:
#         action = np.array([1, 1, 1]) * (box_pos - object_pos)
#         grip = 1.0
#         action *= 3.0
#         holding = True
#         rotate_bowl = True
#         print("action", action)
#         print("Moving to Box")
#         print("np.linalg.norm((box_pos - object_pos)[:2])", np.linalg.norm((box_pos - object_pos)[:2]))
#     else:
#         action = np.zeros((3,))
#         grip=0.
#         holding = True
#         print('Dropping')



#     action = np.append(action, [grip])

#     if args.save_video:
#         img = env.render()
#         images.append(img)

#     time.sleep(0.05)
#     o, r, d, info = env.step(action)
#     print(action)
#     print(o[3])
#     print(r)
#     print('object to goal: {}'.format(info['object_goal_distance']))
#     print('object to gripper: {}'.format(info['object_gripper_distance']))
#     episode_reward += r
#     ts += 1
#     # reward_per_ts.append(r)

holding = False
rotate_object = False

# for i in range(1000):
#     ee_pos = np.array(env.get_end_effector_pos())
#     lego_pos = np.array(env.get_object_midpoint(obj_key))
#     box_pos = np.array(env.get_object_midpoint('box'))
#     # handle_l_id = bullet.get_index_by_attribute(env._objects['box'], 'link_name', 'handle_l')
#     # handle_l_pos = np.array(bullet.get_link_state(env._objects['box'], handle_l_id, 'pos'))
#     handle_r_id = bullet.get_index_by_attribute(env._objects['box'], 'link_name', 'handle_r')
#     handle_r_pos = np.array(bullet.get_link_state(env._objects['box'], handle_r_id, 'pos'))
#     lid_id = bullet.get_index_by_attribute(env._objects['box'], 'link_name', 'lid')
#     lid_pos = np.array(bullet.get_link_state(env._objects['box'], handle_r_id, 'pos'))

#     # print("handle_l_pos", handle_l_pos)
#     print("box_pos", box_pos)

#     # Door angle calc
#     door_angle = env.get_door_angle()
#     print("door_angle", door_angle)
#     opened = abs(door_angle - (np.pi / 2)) < 0.11
#     lost_handle_grip = False # it will never lose grip of the lego
#     handle_pos = handle_r_pos
#     object_pos = box_pos
#     print("lost_handle_grip", lost_handle_grip)
#     xyz_diff = object_pos - ee_pos
#     xy_diff = xyz_diff[:2]
#     xy_goal_diff = (env._goal_pos - object_pos)[:2]
#     base_angle_diff = abs(utils.angle(object_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0])))
#     print("base_angle_diff", base_angle_diff)

#     a = utils.angle(ee_pos[:2], np.array([0.7, 0]))
#     if door_angle > 1.2:
#         if i % 5 == 0:
#             action = np.array([0.7, 0, 1]) - ee_pos
#         else:
#             action = np.array([math.sin(a), -math.cos(a), 0.1])
#     elif door_angle > 0.2:
#         action = lid_pos - ee_pos
#         action /= 3
#     else:
#         break
#     grip = 1.

#     action = np.append(action, [grip])

#     if args.save_video:
#         img = env.render()
#         images.append(img)

#     time.sleep(0.05)
#     o, r, d, info = env.step(action)
#     print(action)
#     print(o[3])
#     print(r)
#     print('object to goal: {}'.format(info['object_goal_distance']))
#     print('object to gripper: {}'.format(info['object_gripper_distance']))
#     episode_reward += r
#     ts += 1
#     # reward_per_ts.append(r)

def plot_reward(rewards):
	import matplotlib.pyplot as plt
	plt.plot(list(range(len(rewards))), rewards, color='lightskyblue')
	plt.title("Scripted Door opening Reward vs. Timestep")
	plt.xlabel("Timestep")
	plt.ylabel("Reward")
	plt.savefig('scripted_door_reward.png')
	plt.close()

print('Episode reward: {}'.format(episode_reward))
object_pos = env.get_object_midpoint(obj_key)
if object_pos[2] > -0.1:
    num_grasps += 1

if args.save_video:
    utils.save_video('data/lego_test_{}.avi'.format(0), images)

plot_reward(reward_per_ts)
