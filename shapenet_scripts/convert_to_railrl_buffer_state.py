import pickle
import os.path as osp
import numpy as np
from tqdm import tqdm

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
import roboverse


# INPUT_BUFFER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'
#                 'feb25_SawyerReach-v0_state_2K_dense_reward_randomize_noise_std'
#                 '_0.3/combined_all_2020-02-25T11-52-48.pkl')
# OUTPUT_NAME = 'railrl_obs_dict_buffer_100K.pkl'

INPUT_BUFFER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'
                'sac_runs_feb11/env-11Feb2020-af888d54-seed=527_2020-02-11_20-'
                '11-04uk2zf5y4/consolidated_buffer.pkl')
OUTPUT_NAME = 'railrl_grasping_buffer_state_1M.pkl'
INPUT_BUFFER = '/media/avi/data/Work/github/jannerm/bullet-manipulation/data/feb12_edgefix_state_2K_dense_reward_randomize/combined_all_2020-02-12T17-02-34.pkl'
OUTPUT_NAME = 'railrl_scriped_grasping_500K.pkl'

if __name__ == "__main__":
    input_buffer = pickle.load(open(INPUT_BUFFER, 'rb'))
    output_dir = osp.dirname(INPUT_BUFFER)
    output_filename = osp.join(output_dir, OUTPUT_NAME)

    input_buffer_size = len(input_buffer['actions'])
    env = roboverse.make('SawyerReach-v0', gui=False, randomize=True,
                         observation_mode='state', reward_type='shaped',
                         transpose_image=True)
    output_buffer = EnvReplayBuffer(input_buffer_size, env)

    path_length = 50
    assert input_buffer_size % path_length == 0
    num_traj = int(input_buffer_size/path_length)

    for i in tqdm(range(num_traj)):
        start_index = i*path_length
        end_index = i*path_length + path_length
        path = dict(
            actions=np.asarray(input_buffer['actions'][start_index:end_index]),
            rewards=np.asarray(input_buffer['rewards'][start_index:end_index]),
            terminals=np.asarray(
                input_buffer['terminals'][start_index:end_index]),
            observations=np.asarray(
                input_buffer['observations'][start_index:end_index]),
            next_observations=np.asarray(
                input_buffer['next_observations'][start_index:end_index]),
            agent_infos=np.asarray([{} for i in range(path_length)]),
            env_infos=np.asarray([{} for i in range(path_length)]),
        )
        output_buffer.add_path(path)
    input_buffer = None
    pickle.dump(output_buffer, open(output_filename, 'wb'))