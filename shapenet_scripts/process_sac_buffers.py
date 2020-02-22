import pickle
import numpy as np
import os
import gzip


BUFFER_FOLDER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'
                 'sac_runs_feb11/env-11Feb2020-af888d54-seed=527_2020-02-11_20-11-04uk2zf5y4')
OUTPUT_NAME = os.path.join(BUFFER_FOLDER, 'consolidated_buffer.pkl')


def main():
    obs_keys = ['observations', 'next_observations']
    non_obs_keys = ['actions', 'rewards', 'terminals']
    all_keys = obs_keys + non_obs_keys
    all_data = {
        'observations': [],
        'next_observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
    }

    checkpoints = list(range(100, 1001, 100))
    for checkpoint in checkpoints:
        filename = os.path.join(BUFFER_FOLDER,
                                'checkpoint_{}'.format(checkpoint),
                                'replay_pool.pkl')

        with gzip.open(filename, 'rb') as f:
            latest_samples = pickle.load(f)

        for key in obs_keys:
            all_data[key].append(latest_samples[key]['observations'])

        for key in non_obs_keys:
            all_data[key].append(latest_samples[key])

    for key in all_keys:
        all_data[key] = np.concatenate(all_data[key], axis=0)

    pickle.dump(all_data, open(OUTPUT_NAME, 'wb+'))


if __name__ == '__main__':
    main()