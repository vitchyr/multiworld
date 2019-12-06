import argparse
from distutils.util import strtobool
import json
import os
import pickle
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from softlearning.misc.utils import save_video
from softlearning.environments.adapters.gym_adapter import GymAdapter

import roboverse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--num-trajectories", type=int, default=2000)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--gui", dest="gui", action="store_true", default=False)

    args = parser.parse_args()

    return args


def simulate_policy(args):
    session = tf.keras.backend.get_session()
    checkpoint_path = args.checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    evaluation_environment = GymAdapter(
        env=roboverse.make("SawyerGraspOne-v0", gui=args.gui))

    policy = (
        get_policy_from_variant(variant, evaluation_environment))
    policy.set_weights(picklable['policy_weights'])

    timestamp = roboverse.utils.timestamp()
    data_save_path = os.path.join(__file__, "../..", 'data',
                                  args.data_save_directory, timestamp)
    data_save_path = os.path.abspath(data_save_path)
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    pool = roboverse.utils.DemoPool()
    num_grasps = 0

    with policy.set_deterministic(False):
        for _ in tqdm(range(args.num_trajectories)):
            observation = evaluation_environment.reset()
            for i in range(args.num_timesteps):
                action = policy.actions_np(np.expand_dims(
                    observation['observations'], axis=0))
                action_noisy = action[0] + \
                               np.random.normal(scale=args.noise_std)
                next_state, reward, done, info = \
                    evaluation_environment.step(action_noisy)
                pool.add_sample(
                    observation['observations'],
                    action_noisy,
                    next_state['observations'],
                    reward,
                    done)
                observation = next_state
                if args.gui:
                    time.sleep(0.1)

            if info['object_goal_distance'] < 0.05:
                num_grasps += 1
                print('Num grasps: {}'.format(num_grasps))

    params = evaluation_environment.get_params()
    pool.save(params, data_save_path,
              '{}_pool_{}.pkl'.format(timestamp, pool.size))

if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
