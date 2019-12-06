import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time

import roboverse

OBJECT_NAME = 'lego'


def scripted_non_markovian(env, pool, render_images):
    env.reset()
    target_pos = env.get_object_midpoint(OBJECT_NAME)
    target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
    # the object is initialized above the table, so let's compensate for it
    target_pos[2] += -0.05
    images = []

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()

        if i < 25:
            action = target_pos - ee_pos
            action[2] = 0.
            action *= 5.0
            grip = 0.
        elif i < 35:
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 5.0
            grip = 0.
        elif i < 42:
            action = np.zeros((3,))
            grip = 0.5
        else:
            action = np.zeros((3,))
            action[2] = 1.0
            grip = 1.

        action = np.append(action, [grip])

        if render_images:
            img = env.render()
            images.append(Image.fromarray(np.uint8(img)))

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)

    success = info['object_goal_distance'] < 0.05
    return success, images


def scripted_markovian(env, pool, render_images):
    observation = env.reset()
    target_pos = env.get_object_midpoint(OBJECT_NAME)
    target_pos[:2] += np.random.uniform(low=-0.05, high=0.05, size=(2,))
    target_pos[2] += np.random.uniform(low=-0.01, high=0.01, size=(1,))
    # the object is initialized above the table, so let's compensate for it
    # target_pos[2] += -0.01
    images = []

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        xyz_diff = target_pos - ee_pos
        xy_diff = xyz_diff[:2]
        # print(observation[3])
        # print(xyz_diff)

        if np.linalg.norm(xyz_diff) > 0.02 and observation[3] > 0.025:
            action = target_pos - ee_pos
            action *= 5.0
            if np.linalg.norm(xy_diff) > 0.05:
                action[2] *= 0.5
            grip = -1.0
            # print('Approaching')
        elif observation[3] > 0.025:
            # o[3] is gripper tip distance
            action = np.zeros((3,))
            grip = 1.
            # print('Grasping')
        elif info['gripper_goal_distance'] > 0.01:
            action = env._goal_pos - ee_pos
            action *= 5.0
            grip = 1.
            # print('Moving')
        else:
            action = np.zeros((3,))
            grip = 1.
            # print('Holding')

        action = np.append(action, [grip])
        action += np.random.normal(scale=args.noise_std)

        if render_images:
            img = env.render()
            images.append(Image.fromarray(np.uint8(img)))

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)
        # time.sleep(0.2)
        observation = next_state

    success = info['object_goal_distance'] < 0.05
    return success, images


def main(args):

    timestamp = roboverse.utils.timestamp()
    data_save_path = os.path.join(__file__, "../..", 'data',
                                  args.data_save_directory, timestamp)
    data_save_path = os.path.abspath(data_save_path)
    video_save_path = os.path.join(data_save_path, "videos")
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
        os.makedirs(video_save_path)

    env = roboverse.make('SawyerGraspOne-v0', gui=args.gui)
    num_grasps = 0
    pool = roboverse.utils.DemoPool()

    for j in tqdm(range(args.num_trajectories)):
        render_images = args.video_save_frequency > 0 and \
                        j % args.video_save_frequency == 0

        if args.non_markovian:
            success, images = scripted_non_markovian(env, pool, render_images)
        else:
            success, images = scripted_markovian(env, pool, render_images)

        if success:
            num_grasps += 1
            print(num_grasps)

        if render_images:
            images[0].save('{}/{}.gif'.format(video_save_path, j),
                           format='GIF', append_images=images[1:],
                           save_all=True, duration=100, loop=0)

    params = env.get_params()
    pool.save(params, data_save_path,
              '{}_pool_{}.pkl'.format(timestamp, pool.size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("-n", "--num-trajectories", type=int, default=2000)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--video_save_frequency", type=int,
                        default=0, help="Set to zero for no video saving")
    parser.add_argument("--gui", dest="gui", action="store_true", default=False)
    parser.add_argument("--non-markovian", dest="non_markovian",
                        action="store_true", default=False)

    args = parser.parse_args()

    main(args)