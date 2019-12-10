import argparse
import time
import subprocess

def get_data_save_directory(args):
    data_save_directory = args.data_save_directory

    if args.num_trajectories > 1000:
        data_save_directory += '_{}K'.format(int(args.num_trajectories/1000))
    else:
        data_save_directory += '_{}'.format(args.num_trajectories)

    if args.sparse:
        data_save_directory += '_sparse_reward'
    else:
        data_save_directory += '_dense_reward'

    if args.randomize:
        data_save_directory += '_randomize'
    else:
        data_save_directory += '_fixed_position'

    return data_save_directory


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("-n", "--num-trajectories", type=int, default=2000)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=10)
    parser.add_argument("--sparse", dest="sparse", action="store_true",
                        default=False)
    parser.add_argument("--randomize", dest="randomize", action="store_true",
                        default=False)
    args = parser.parse_args()

    num_trajectories_per_thread = int(
        args.num_trajectories / args.num_parallel_threads)
    if args.num_trajectories % args.num_parallel_threads != 0:
        num_trajectories_per_thread += 1
    save_directory = get_data_save_directory(args)
    command = ['python',
               'shapenet_scripts/randomized_scripted_grasping.py',
               '-d{}'.format(save_directory),
               '-n {}'.format(num_trajectories_per_thread),
               '-p {}'.format(args.num_parallel_threads),
               ]
    if args.sparse:
        command.append('--sparse')
    if args.randomize:
        command.append('--randomize')

    subprocesses = []
    for i in range(args.num_parallel_threads):
        subprocesses.append(subprocess.Popen(command))
        time.sleep(1)

    exit_codes = [p.wait() for p in subprocesses]
    subprocess.call(['python',
                     'shapenet_scripts/combine_trajectories.py',
                     '-d{}'.format(save_directory)]
                    )
    # print(exit_codes)
