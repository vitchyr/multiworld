import pickle
import os.path as osp
import numpy as np
from tqdm import tqdm

from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
import roboverse

# Buffer for scripted reaching from pixels
# INPUT_BUFFER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'
#                 'feb26_SawyerReach-v0_pixels_2K_dense_reward_randomize_noise'
#                 '_std_0.3/combined_all_2020-02-26T18-52-21.pkl')
# OUTPUT_NAME = 'railrl_obs_dict_buffer_100K.pkl'

# INPUT_BUFFER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'
#                 'sac_runs_feb11/env-11Feb2020-af888d54-seed='
#                 '527_2020-02-11_20-11-04uk2zf5y4/pixels_buffer/'
#                 'consolidated_buffer_pixel_only.pkl')
# OUTPUT_NAME = 'railrl_obs_dict_grasping_buffer_1M.pkl'

INPUT_BUFFER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'\
               'april10_SawyerGraspOne-v0_pixels_5K_dense_reward_randomize_noi'
                'se_std_0.1/combined_all_2020-04-10T14-07-56.pkl')
OUTPUT_NAME = 'railrl_obs_dict_scripted_grasping_250K.pkl'


if __name__ == "__main__":
    input_buffer = pickle.load(open(INPUT_BUFFER, 'rb'))
    output_dir = osp.dirname(INPUT_BUFFER)
    output_filename = osp.join(output_dir, OUTPUT_NAME)

    input_buffer_size = len(input_buffer['actions'])
    env = roboverse.make('SawyerReach-v0', gui=False, randomize=True,
                         observation_mode='pixels', reward_type='shaped',
                         transpose_image=True)
    output_buffer = ObsDictReplayBuffer(input_buffer_size, env, observation_key='image')

    path_length = 50
    assert input_buffer_size % path_length == 0
    num_traj = int(input_buffer_size/path_length)

    for i in tqdm(range(num_traj)):
        start_index = i*path_length
        end_index = i*path_length + path_length
        path = dict(
            actions=np.asarray(input_buffer['actions'][start_index:end_index]),
            rewards=np.asarray(input_buffer['rewards'][start_index:end_index]),
            terminals=np.asarray(input_buffer['terminals'][start_index:end_index]),
        )

        obs_processed = []
        next_obs_processed = []
        for j in range(start_index, end_index):
            image = input_buffer['observations'][j][0]['image']
            # FIXME(avi) You know what you've done.
            if len(image.shape) == 1:
                assert image.shape[0] == 48*48*3
                image = image.reshape((48, 48, 3))
                image = image*255.0
                image = image.astype(np.uint8)

            image = np.transpose(image, (2, 0, 1))
            image = image.flatten()
            image = np.float32(image)/255.0
            state = input_buffer['observations'][j][0]['state']
            obs_processed.append(dict(image=image, state=state))

            next_image = input_buffer['next_observations'][j][0]['image']
            # FIXME(avi) Same as above.

            if len(next_image.shape) == 1:
                next_image = next_image.reshape((48, 48, 3))
                next_image = next_image*255.0
                next_image = next_image.astype(np.uint8)

            next_image = np.transpose(next_image, (2, 0, 1))
            next_image = next_image.flatten()
            next_image = np.float32(next_image) / 255.0
            next_state = input_buffer['next_observations'][j][0]['state']
            next_obs_processed.append(dict(image=next_image, state=next_state))

        path['observations'] = obs_processed
        path['next_observations'] = next_obs_processed

        output_buffer.add_path(path)

    input_buffer = None
    pickle.dump(output_buffer, open(output_filename, 'wb'), protocol=4)
