import roboverse
import numpy as np

from tqdm import tqdm
import pickle
import os
from PIL import Image
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("save_directory", type=str)
parser.add_argument("--num_trajectories", type=int, default=2000)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")
args = parser.parse_args()

env = roboverse.make('SawyerGraspOne-v0', gui=False)
object_name = 'lego'

num_grasps = 0
image_data = []

obs_dim = env.observation_space.shape
assert(len(obs_dim) == 1)
obs_dim = obs_dim[0]
act_dim = env.action_space.shape[0]

dataset = {
    'observations':  np.zeros(
        (args.num_trajectories, args.num_timesteps, obs_dim)),
    'next_observations': np.zeros(
        (args.num_trajectories, args.num_timesteps, obs_dim)),
    'actions': np.zeros(
        (args.num_trajectories, args.num_timesteps, act_dim)),
    'rewards': np.zeros(
        (args.num_trajectories, args.num_timesteps)),
}

if not os.path.exists(args.save_directory):
    os.makedirs(args.save_directory)
time_string = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
gif_dir = os.path.join(args.save_directory, time_string + '_gifs')
os.makedirs(gif_dir)

for j in tqdm(range(args.num_trajectories)):
    env.reset()
    target_pos = env.get_object_midpoint(object_name)
    target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
    # the object is initialized above the table, so let's compensate for it
    target_pos[2] += -0.05
    images = []

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()

        if i < 25:
            action = target_pos - ee_pos
            action[2] = 0.
            action *= 3.0
            grip = 0.
        elif i < 35:
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 2.0
            grip = 0.
        elif i < 42:
            action = np.zeros((3,))
            grip = 0.5
        else:
            action = np.zeros((3,))
            action[2] = 1.0
            grip = 1.

        action = np.append(action, [grip])

        if args.video_save_frequency > 0 and j % args.video_save_frequency == 0:
            img = env.render()
            images.append(Image.fromarray(np.uint8(img)))

        dataset['observations'][j, i] = env.get_observation()
        next_state, reward, done, info = env.step(action)
        dataset['next_observations'][j, i] = next_state
        dataset['actions'][j, i] = action
        dataset['rewards'][j, i] = reward

    object_pos = env.get_object_midpoint(object_name)
    if object_pos[2] > -0.1:
        num_grasps += 1
        print('Num grasps: {}'.format(num_grasps))

    if args.video_save_frequency > 0 and j % args.video_save_frequency == 0:
        images[0].save('{}/{}.gif'.format(gif_dir, j),
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)

save_path = os.path.join(args.save_directory, '{}.pkl'.format(time_string))
with open(save_path, 'wb+') as fp:
    pickle.dump(dataset, fp)
