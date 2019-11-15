import pickle
import argparse
import roboverse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_directory", type=str)
parser.add_argument("--output_directory", type=str)
args = parser.parse_args()

args.data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'SawyerGrasp', args.data_directory)
args.output_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'SawyerGrasp', args.output_directory)

trajectories = []
timestamp = roboverse.utils.timestamp()

for root, dirs, files in os.walk(args.data_directory):
    for f in files:
        if "pool" in f:
            with open(os.path.join(args.data_directory, f), 'rb') as fp:
                t = pickle.load(fp)

            trajectories.extend(t)

if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)

with open(os.path.join(args.output_directory, 'data_{0}.pkl'.format(timestamp)), 'wb+') as fp:
    pickle.dump(trajectories, fp)
