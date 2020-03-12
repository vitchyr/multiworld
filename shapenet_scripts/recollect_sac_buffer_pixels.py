import pickle
import os
import argparse
from tqdm import tqdm

import roboverse
import roboverse.bullet as bullet

BUFFER_FOLDER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'
                 'sac_runs_feb11/env-11Feb2020-af888d54-seed=527_2020-02-11_20-11-04uk2zf5y4')
INPUT_NAME = os.path.join(BUFFER_FOLDER, 'consolidated_buffer.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-trajectories", type=int)
    parser.add_argument("--offset", type=int)
    parser.add_argument("-e", "--env", type=str, default='SawyerGraspOne-v0',
                        choices=('SawyerGraspOne-v0', 'SawyerReach-v0'))

    args = parser.parse_args()

    with open(INPUT_NAME, 'rb') as f:
        entire_buffer = pickle.load(f)
    env = roboverse.make(args.env, reward_type='shaped',
                         randomize=True, observation_mode='pixels_debug')
    trajectory_size = 50
    dataset_size = entire_buffer['actions'].shape[0]
    assert dataset_size%trajectory_size == 0
    path_returns = []
    offset = 0

    pool = roboverse.utils.DemoPool()

    for i in tqdm(range(args.num_trajectories)):
        obs = env.reset()
        start_ind = i*trajectory_size + args.offset
        # print(start_ind)
        init_obs = entire_buffer['observations'][start_ind]
        bullet.set_body_state(env._objects['lego'], init_obs[4:7], init_obs[7:])
        path_return = 0

        for j in range(trajectory_size):
            action = entire_buffer['actions'][start_ind+j]
            next_obs, reward, done, info = env.step(action)
            path_return += reward
            pool.add_sample(obs, action, next_obs, reward, done)
            obs = next_obs

        path_returns.append(path_return)

    params = env.get_params()
    data_save_path = os.path.join(BUFFER_FOLDER, 'pixels_buffer')
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    pool.save(params, data_save_path,
              '{}_{}_pool.pkl'.format(args.offset, args.offset + pool.size))