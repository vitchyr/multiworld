import roboverse
import numpy as np
from tqdm import tqdm
import os
from roboverse.bullet.misc import quat_to_deg 

from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=1000)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")
parser.add_argument("--gui", dest="gui", action="store_true", default=False)

args = parser.parse_args()
data_save_path = "/home/ashvin/data/sasha/demos/" + args.name + ".npy"
video_save_path = "/home/ashvin/data/sasha/demos/videos"

env = roboverse.make('SawyerRigMultiobj-v0', gui=args.gui)
object_name = 'obj'
num_grasps = 0
image_data = []

act_dim = env.action_space.shape[0]
imlength = env.obs_img_dim * env.obs_img_dim * 3

if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
    os.makedirs(video_save_path)

dataset = {
    'observations': np.zeros((args.num_trajectories, args.num_timesteps, imlength), dtype=np.uint8),
    'actions': np.zeros((args.num_trajectories, args.num_timesteps, act_dim), dtype=np.float),
    'env': np.zeros((args.num_trajectories, imlength), dtype=np.uint8),
    }

for i in tqdm(range(args.num_trajectories)):
    if i % 10 == 0:
        noise = 1
    else:
        noise = 0.01

    env.reset()
    target_pos = env.get_object_midpoint(object_name)
    images = []

    dataset['env'][i, :] = np.uint8(env.render_obs().transpose()).flatten()
    for j in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        dataset['observations'][i, j, :] = img.transpose().flatten()
        images.append(Image.fromarray(img))

        ee_pos = np.append(env.get_end_effector_pos(), quat_to_deg(env.theta)[2] / 180)
        target_pos = np.append(env.get_object_midpoint('obj'), env.get_object_deg()[2] / 180)
        target_pos[1] += 0.0065
        if j < 20:
            action = target_pos - ee_pos
            action[2] = 0.
            action *= 3.0
            grip = -1.
        elif j < 35:
            action = target_pos - ee_pos
            action *= 3.0
            action[2] *= 2.0
            grip = -1.
        elif j < 42:
            action = np.zeros((4,))
            grip = 1.0
        else:
            action = np.zeros((4,))
            action[2] = 1.0
            action = np.random.normal(action, 0.25)
            grip = 1.

        action = np.append(action, [grip])
        action = np.random.normal(action, noise)
        action = np.clip(action, a_min=-1, a_max=1)


        dataset['actions'][i, j, :] = action
        observation = env.get_observation()
        next_observation, reward, done, info = env.step(action)


    object_pos = env.get_object_midpoint(object_name)
    if object_pos[2] > -0.33:
        num_grasps += 1
    

    if args.video_save_frequency > 0 and i % args.video_save_frequency == 0:
        images[0].save('{}/{}.gif'.format(video_save_path, i),
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)

print('Success Rate: {}'.format(num_grasps / args.num_trajectories))
np.save(data_save_path, dataset)