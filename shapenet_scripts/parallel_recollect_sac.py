import subprocess
import time
import os
import pickle

BUFFER_FOLDER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'
                 'sac_runs_feb11/env-11Feb2020-af888d54-seed=527_2020-02-11_20-11-04uk2zf5y4')


if __name__ == "__main__":
    num_traj_per_thread = 2000
    num_threads = 10

    # num_traj_per_thread = int(10)
    # num_threads = 2

    traj_len = 50

    offsets = [i*num_traj_per_thread*traj_len for i in range(num_threads)]
    commands = []
    for i in range(num_threads):
        command = ['python',
                   'shapenet_scripts/recollect_sac_buffer_pixels.py',
                   '-n',
                   str(num_traj_per_thread),
                   '--offset',
                   str(offsets[i]),
                   ]
        commands.append(command)

    subprocesses = []
    for i in range(num_threads):
        subprocesses.append(subprocess.Popen(commands[i]))
        time.sleep(1)
    exit_codes = [p.wait() for p in subprocesses]
    print('Exit codes: {}'.format(exit_codes))


    keys = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
    fields_all = {}
    for key in keys:
        fields_all[key] = []

    for i in range(len(offsets)):
        filename = '{}_{}_pool.pkl'.format(offsets[i], offsets[i] + num_traj_per_thread*traj_len)
        filepath = os.path.join(BUFFER_FOLDER, 'pixels_buffer', filename)

        with open(filepath, 'rb') as fp:
            trajectories = pickle.load(fp)

        for key in keys:
            fields_all[key].extend(trajectories[key])

    save_all_path = os.path.join(BUFFER_FOLDER, 'pixels_buffer', 'consolidated_buffer.pkl')
    with open(save_all_path, 'wb+') as fp:
        pickle.dump(fields_all, fp)
        print('saved to {}'.format(save_all_path))
