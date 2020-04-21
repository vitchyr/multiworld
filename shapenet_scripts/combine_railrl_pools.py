import pickle
import argparse
import roboverse
import os
import os.path as osp

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

EXTRA_POOL_SPACE = int(1e5)
REWARD_NEGATIVE = -10
REWARD_POSITIVE = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str,
                        choices=('SawyerGraspV2-v0', 'SawyerGraspOneV2-v0',
                                 'SawyerGraspTenV2-v0'))
    parser.add_argument("-d", "--data-directory", type=str, required=True)
    parser.add_argument("-o", "--observation-mode", type=str, default='state',
                        choices=('state', 'pixels', 'pixels_debug'))
    args = parser.parse_args()

    data_directory = osp.join(
        os.path.dirname(__file__), "..", 'data', args.data_directory)
    print(data_directory)
    # keys = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
    timestamp = roboverse.utils.timestamp()

    pools = []
    success_pools = []

    for root, dirs, files in os.walk(data_directory):
        for f in files:
            if "pool" in f:
                with open(os.path.join(root, f), 'rb') as fp:
                    pool = pickle.load(fp)
                if 'success_only' in f:
                    success_pools.append(pool)
                else:
                    pools.append(pool)

    original_pool_size = 0
    for pool in pools:
        original_pool_size += pool._top
    pool_size = original_pool_size + EXTRA_POOL_SPACE

    env = roboverse.make(args.env,
                         observation_mode=args.observation_mode,
                         transpose_image=True)
    if args.observation_mode == 'state':
        consolidated_pool = EnvReplayBuffer(pool_size, env)
        for pool in pools:
            for i in range(pool._top):

                # if pool._rewards[i] < 0:
                #     reward_corrected = REWARD_NEGATIVE
                # elif pool._rewards[i] > 0:
                #     reward_corrected = REWARD_POSITIVE
                # else:
                #     raise ValueError

                reward_corrected = pool._rewards[i]
                consolidated_pool.add_sample(
                    observation=pool._observations[i],
                    action=pool._actions[i],
                    reward=reward_corrected,
                    next_observation=pool._next_obs[i],
                    terminal=pool._terminals[i],
                    env_info={},
                )

    elif args.observation_mode in ['pixels', 'pixels_debug']:
        obs_key = 'image'
        consolidated_pool = ObsDictReplayBuffer(pool_size, env,
                                                observation_key=obs_key)
        for pool in pools:
            # import IPython; IPython.embed()
            path = dict(
                rewards=[],
                actions=[],
                terminals=[],
                observations=[],
                next_observations=[],
            )
            for i in range(pool._top):
                path['rewards'].append(pool._rewards[i])
                path['actions'].append(pool._actions[i])
                path['terminals'].append(pool._terminals[i])
                path['observations'].append(dict(image=pool._obs[obs_key][i]))
                path['next_observations'].append(dict(image=pool._next_obs[obs_key][i]))

                if pool._terminals[i]:
                    consolidated_pool.add_path(path)
                    path = dict(
                        rewards=[],
                        actions=[],
                        terminals=[],
                        observations=[],
                        next_observations=[],
                    )

            if len(path['rewards']) > 0:
                consolidated_pool.add_path(path)

    else:
        raise NotImplementedError

    path = osp.join(os.path.dirname(__file__), "..", 'data',
                    args.data_directory, 'railrl_consolidated.pkl')
    pickle.dump(consolidated_pool, open(path, 'wb'), protocol=4)
