import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import argparse
import pickle

import roboverse
import skvideo.io

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

OBJECT_NAME = 'lego'
EPSILON = 0.05
V2_GRASPING_ENVS = ['SawyerGraspV2-v0',
                    'SawyerGraspTenV2-v0',
                    'SawyerGraspOneV2-v0']


def scripted_non_markovian_grasping(env, pool, render_images):
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
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        if render_images:
            img = env.render()
            images.append(img)

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)

    success = info['object_goal_distance'] < 0.05
    return success, images


def scripted_markovian_grasping(env, pool, render_images):
    observation = env.reset()
    if args.randomize:
        target_pos = np.random.uniform(low=env._object_position_low,
                                      high=env._object_position_high)
        target_pos[:2] += np.random.uniform(low=-0.03, high=0.03, size=(2,))
        target_pos[2] += np.random.uniform(low=-0.02, high=0.02, size=(1,))
    else:
        target_pos = env.get_object_midpoint(OBJECT_NAME)
        target_pos[:2] += np.random.uniform(low=-0.05, high=0.05, size=(2,))
        target_pos[2] += np.random.uniform(low=-0.01, high=0.01, size=(1,))

    # the object is initialized above the table, so let's compensate for it
    # target_pos[2] += -0.01
    images = []
    grip_open = -0.8
    grip_close = 0.8

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        xyz_diff = target_pos - ee_pos
        xy_diff = xyz_diff[:2]
        # print(observation[3])
        # print(xyz_diff)
        if isinstance(observation, dict):
            gripper_tip_distance = observation['state'][3]
        else:
            gripper_tip_distance = observation[3]

        if np.linalg.norm(xyz_diff) > 0.02 and gripper_tip_distance > 0.025:
            action = target_pos - ee_pos
            action *= 5.0
            if np.linalg.norm(xy_diff) > 0.05:
                action[2] *= 0.5
            grip = grip_open
            # print('Approaching')
        elif gripper_tip_distance > 0.025:
            # o[3] is gripper tip distance
            action = np.zeros((3,))
            if grip == grip_open:
                grip = 0.
            else:
                grip = grip_close
            # print('Grasping')
        elif info['gripper_goal_distance'] > 0.01:
            action = env._goal_pos - ee_pos
            action *= 5.0
            grip = grip_close
            # print('Moving')
        else:
            action = np.zeros((3,))
            grip = grip_close
            # print('Holding')

        action = np.append(action, [grip])
        action += np.random.normal(scale=args.noise_std)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        if render_images:
            img = observation['image']
            images.append(img)

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)
        # time.sleep(0.2)
        observation = next_state

    success = info['object_goal_distance'] < 0.05
    return success, images


def scripted_grasping_V2(env, pool, success_pool, random_actions=False):
    """

    :param env:
    :param pool:
    :param success_pool:
    :param random_actions: When set to True, executes random actions instead
    of following the object(s).
    :return:
    """

    observation = env.reset()
    object_ind = np.random.randint(0, env._num_objects)
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    for _ in range(args.num_timesteps):

        if not random_actions:
            if isinstance(observation, dict):
                object_pos = observation['state'][
                             object_ind*7 + 8: object_ind*7 + 8 + 3]
                ee_pos = observation['state'][:3]
            else:
                object_pos = observation[
                             object_ind * 7 + 8: object_ind * 7 + 8 + 3]
                ee_pos = observation[:3]

            action = object_pos - ee_pos
            action = action*4.0
            action += np.random.normal(scale=args.noise_std, size=(3,))
        else:
            action = np.random.uniform(low=-1.0, high=1.0, size=(3,))
            if np.random.uniform() < 0.9:
                action[2] = -1

        theta_action = np.random.uniform(low=-1 + EPSILON, high=1-EPSILON)
        action = np.concatenate((action, np.asarray([theta_action])))

        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
        next_observation, reward, done, info = env.step(action)

        actions.append(action)
        observations.append(observation)
        rewards.append(reward)
        terminals.append(done)
        infos.append(info)
        next_observations.append(next_observation)

        observation = next_observation

        if done:
            break

    path = dict(
        actions=actions,
        rewards=np.asarray(rewards).reshape((-1, 1)),
        terminals=np.asarray(terminals).reshape((-1, 1)),
        infos=infos,
        observations=observations,
        next_observations=next_observations,
    )

    if not isinstance(observation, dict):
        path_length = len(rewards)
        path['agent_infos'] = np.asarray([{} for i in range(path_length)])
        path['env_infos'] = np.asarray([{} for i in range(path_length)])

    pool.add_path(path)
    if rewards[-1] > 0:
        success_pool.add_path(path)


def scripted_markovian_reaching(env, pool, render_images):
    observation = env.reset()
    if args.randomize:
        target_pos = np.random.uniform(low=env._object_position_low,
                                      high=env._object_position_high)
        target_pos[:2] += np.random.uniform(low=-0.03, high=0.03, size=(2,))
        target_pos[2] += np.random.uniform(low=-0.02, high=0.02, size=(1,))
    else:
        target_pos = env.get_object_midpoint(OBJECT_NAME)
        target_pos[:2] += np.random.uniform(low=-0.05, high=0.05, size=(2,))
        target_pos[2] += np.random.uniform(low=-0.01, high=0.01, size=(1,))

    images = []

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        xyz_diff = target_pos - ee_pos
        xy_diff = xyz_diff[:2]

        action = target_pos - ee_pos
        action *= 5.0
        if np.linalg.norm(xy_diff) > 0.05:
            action[2] *= 0.5
        grip = 0.0
        action = np.append(action, [grip])
        action += np.random.normal(scale=args.noise_std)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        if render_images:
            img = observation['image']
            images.append(img)

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)
        # time.sleep(0.2)
        observation = next_state

    success = info['object_gripper_distance'] < 0.03
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

    reward_type = 'sparse' if args.sparse else 'shaped'
    if args.env in V2_GRASPING_ENVS:
        tranpose_image = True
    env = roboverse.make(args.env, reward_type=reward_type,
                         gui=args.gui, randomize=args.randomize,
                         observation_mode=args.observation_mode,
                         transpose_image=tranpose_image)

    num_success = 0
    if args.env == 'SawyerGraspOne-v0' or args.env == 'SawyerReach-v0':
        pool = roboverse.utils.DemoPool()
        success_pool = roboverse.utils.DemoPool()
    elif args.env in V2_GRASPING_ENVS and 'pixels' in args.observation_mode:
        pool_size = args.num_trajectories*args.num_timesteps
        railrl_pool = ObsDictReplayBuffer(pool_size, env, observation_key='image')
        railrl_success_pool = ObsDictReplayBuffer(pool_size, env, observation_key='image')
    elif args.env in V2_GRASPING_ENVS and args.observation_mode == 'state':
        pool_size = args.num_trajectories*args.num_timesteps
        railrl_pool = EnvReplayBuffer(pool_size, env)
        railrl_success_pool = EnvReplayBuffer(pool_size, env)

    for j in tqdm(range(args.num_trajectories)):
        render_images = args.video_save_frequency > 0 and \
                        j % args.video_save_frequency == 0

        if args.env == 'SawyerGraspOne-v0':
            if args.non_markovian:
                success, images = scripted_non_markovian_grasping(env, pool, render_images)
            else:
                success, images = scripted_markovian_grasping(env, pool, render_images)
        elif args.env == 'SawyerReach-v0':
            success, images = scripted_markovian_reaching(env, pool, render_images)
        elif args.env in V2_GRASPING_ENVS:
            assert not render_images
            success = False
            scripted_grasping_V2(env, railrl_pool, railrl_success_pool,
                                 random_actions=args.random_actions)
        else:
            raise NotImplementedError

        if success:
            num_success += 1
            print('Num success: {}'.format(num_success))
            top = pool._size
            bottom = top - args.num_timesteps
            for i in range(bottom, top):
                success_pool.add_sample(
                    pool._fields['observations'][i],
                    pool._fields['actions'][i],
                    pool._fields['next_observations'][i],
                    pool._fields['rewards'][i],
                    pool._fields['terminals'][i]
                )
        if render_images:
            filename = '{}/{}.mp4'.format(video_save_path, j)
            writer = skvideo.io.FFmpegWriter(
                filename,
                inputdict={"-r": "10"},
                outputdict={
                    '-vcodec': 'libx264',
                })
            num_frames = len(images)
            for i in range(num_frames):
                writer.writeFrame(images[i])
            writer.close()

    if args.env == 'SawyerGraspOne-v0' or args.env == 'SawyerReach-v0':
        params = env.get_params()
        pool.save(params, data_save_path,
                  '{}_pool_{}.pkl'.format(timestamp, pool.size))
        success_pool.save(params, data_save_path,
                          '{}_pool_{}_success_only.pkl'.format(
                              timestamp, pool.size))
    elif args.env in V2_GRASPING_ENVS:
        path = osp.join(data_save_path,
                        '{}_pool_{}.pkl'.format(timestamp, pool_size))
        pickle.dump(railrl_pool, open(path, 'wb'), protocol=4)
        path = osp.join(data_save_path,
                        '{}_pool_{}_success_only.pkl'.format(
                            timestamp, pool_size))
        pickle.dump(railrl_success_pool, open(path, 'wb'), protocol=4)
        print('Num success: {}'.format(np.sum(railrl_success_pool._terminals)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str,
                        choices=('SawyerGraspOne-v0', 'SawyerGraspV2-v0',
                                 'SawyerReach-v0', 'SawyerGraspOneV2-v0',
                                 'SawyerGraspTenV2-v0'))
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("-n", "--num-trajectories", type=int, default=2000)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--video_save_frequency", type=int,
                        default=0, help="Set to zero for no video saving")
    parser.add_argument("--randomize", dest="randomize",
                        action="store_true", default=False)
    parser.add_argument("--random-actions", dest="random_actions",
                        action="store_true", default=False)
    parser.add_argument("--gui", dest="gui", action="store_true", default=False)
    parser.add_argument("--sparse", dest="sparse", action="store_true",
                        default=False)
    parser.add_argument("--non-markovian", dest="non_markovian",
                        action="store_true", default=False)
    parser.add_argument("-o", "--observation-mode", type=str, default='pixels',
                        choices=('state', 'pixels', 'pixels_debug'))

    args = parser.parse_args()

    if args.env in V2_GRASPING_ENVS:
        args.num_timesteps = 20

    main(args)