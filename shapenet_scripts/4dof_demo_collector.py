import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg 
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=5000)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")

args = parser.parse_args()
demo_data_save_path = "/home/ashvin/data/sasha/demos/" + args.name + "_demos"
recon_data_save_path = "/home/ashvin/data/sasha/demos/" + args.name + "_images.npy"
video_save_path = "/home/ashvin/data/sasha/demos/videos"

state_env = roboverse.make('SawyerRigMultiobj-v0', DoF=4)
imsize = state_env.obs_img_dim

renderer_kwargs=dict(
        create_image_format='HWC',
        output_image_format='CWH',
        width=imsize,
        height=imsize,
        flatten_image=True,)

renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
env = InsertImageEnv(state_env, renderer=renderer)
imlength = env.obs_img_dim * env.obs_img_dim * 3

success = 0
returns = 0
act_dim = env.action_space.shape[0]

if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
    os.makedirs(video_save_path)

demo_dataset = []
# recon_dataset = {
#     'observations': np.zeros((args.num_trajectories, args.num_timesteps, imlength), dtype=np.uint8),
#     'object': [],
#     'env': np.zeros((args.num_trajectories, imlength), dtype=np.uint8),
# }

for i in tqdm(range(args.num_trajectories)):
    env.reset()
    trajectory = {
        'observations': [],
        'next_observations': [],
        'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float),
        'rewards': np.zeros((args.num_timesteps), dtype=np.float),
        'terminals': np.zeros((args.num_timesteps), dtype=np.uint8),
        'agent_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'env_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'object_name': env.curr_object,
    }
    images = []
    #recon_dataset['env'][i, :] = np.uint8(env.render_obs().transpose()).flatten()
    #recon_dataset['object'].append(env.curr_object)
    for j in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        #recon_dataset['observations'][i, j, :] = img.transpose().flatten()
        images.append(Image.fromarray(img))

        ee_pos = np.append(env.get_end_effector_pos(), quat_to_deg(env.theta)[2] / 180)
        target_pos = np.append(env.get_object_midpoint('obj'), env.get_object_deg()[2] / 180)
        target_pos[1] += 0.0065
        target_pos[2] -= 0.01

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
            grip = 0.8
        else:
            action = np.zeros((4,))
            action[2] = 1.0
            action = np.random.normal(action, 0.25)
            grip = 1.

        action = np.append(action, [grip])
        action = np.random.normal(action, 0.01)
        action = np.clip(action, a_min=-1, a_max=1)

        observation = env.get_observation()
        next_observation, reward, done, info = env.step(action)

        trajectory['observations'].append(observation)
        trajectory['actions'][j, :] = action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][j] = reward

        returns += reward
    
    success += info['picked_up']

    if args.video_save_frequency > 0 and i % args.video_save_frequency == 0:
        images[0].save('{}/{}.gif'.format(video_save_path, i),
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)

    demo_dataset.append(trajectory)

print('Success Rate: {}'.format(success / args.num_trajectories))
print('Returns: {}'.format(returns / args.num_trajectories))

#np.save(recon_data_save_path, recon_dataset)
step_size = 1000
for i in range(args.num_trajectories // step_size):
    curr_name = demo_data_save_path + '_{0}.pkl'.format(i)
    start_ind, end_ind = i*step_size, (i+1)*step_size
    curr_data = demo_dataset[start_ind:end_ind]
    file = open(curr_name, 'wb')
    pkl.dump(curr_data, file)
    file.close()