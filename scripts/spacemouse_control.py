import roboverse as rv
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from IPython.display import clear_output
import time
plt.ion()

# Variables to define!
DoF = 4
num_timesteps = 50
object_subset = 'test'
# Variables to define!

# Set Up Enviorment
spacemouse = rv.devices.SpaceMouse(DoF=DoF)
state_env = rv.make('SawyerRigMultiobj-v0', gui=True, DoF=DoF, object_subset=object_subset, visualize=False)
imsize = state_env.obs_img_dim

demo_save_path = "/Users/sasha/Desktop/spacemouse/demo_data/{0}_{1}.pkl".format(object_subset, time.time())
recon_save_path = "/Users/sasha/Desktop/spacemouse/recon_data/{0}_{1}.npy".format(object_subset, time.time())
num_trajectories = 0

renderer_kwargs=dict(
	create_image_format='HWC',
	output_image_format='CWH',
	width=imsize,
	height=imsize,
	flatten_image=True)

renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
env = InsertImageEnv(state_env, renderer=renderer)
imlength = env.obs_img_dim * env.obs_img_dim * 3
act_dim = env.action_space.shape[0]
# Set Up Enviorment

def get_empty_traj_dict():
	return {
		'observations': [],
		'next_observations': [],
		'actions': np.zeros((num_timesteps, act_dim), dtype=np.float),
		'rewards': np.zeros((num_timesteps), dtype=np.float),
		'terminals': np.zeros((num_timesteps), dtype=np.uint8),
		'agent_infos': np.zeros((num_timesteps), dtype=np.uint8),
		'env_infos': np.zeros((num_timesteps), dtype=np.uint8),
	}

def get_empty_recon_dict():
	return {
		'observations': [],
		'env': [],
	}

def get_recon_image(env):
	return np.uint8(env.render_obs().transpose()).reshape(1, -1)

def recompute_rewards(trajectory):
	final_state = trajectory['next_observations'][-1]['state_observation']
	for j in range(num_timesteps):
		trajectory['observations'][j]['state_desired_goal'] = final_state
		trajectory['next_observations'][j]['state_desired_goal'] = final_state
		trajectory['rewards'][j] = state_env.compute_reward(
					trajectory['observations'][j],
					trajectory['actions'][j],
					trajectory['next_observations'][j],
					trajectory['next_observations'][j])


def save_datasets(recon_dataset, demo_dataset):
	curr_recon_dataset = {'observations': np.concatenate(recon_dataset['observations'], axis=0),
						'env': np.concatenate(recon_dataset['env'], axis=0)}

	# Save Recon Data
	file = open(recon_save_path, 'wb')
	np.save(file, curr_recon_dataset)
	file.close()
	
	# Save Demo Data
	file = open(demo_save_path, 'wb')
	pkl.dump(demo_dataset, file)
	file.close()

def render():
	clear_output(wait=True)
	img = env.render_obs()
	plt.imshow(img)
	plt.show()
	plt.pause(0.01)

def get_and_process_response():
	response = input(
			'Enter: Add trajectory to both datasets \
			\n D: Add trajectory to demo dataset \
			\n R: Add trajectory to reconstruction dataset \
			\n S: Skip trajectory \
			\n Q: Quit \n')

	add_both = (response == '')
	end = 'Q' in response

	if 'D' in response or add_both:
		# Save To Demo Dataset
		recompute_rewards(trajectory)
		demo_dataset.append(trajectory)
		print('Returns:', sum(trajectory['rewards']))
	
	if 'R' in response or add_both:
		# Save To Reconstruction Dataset
		recon_dataset['env'].append(env_image)
		traj_images = np.concatenate(traj_images, axis=0)
		recon_dataset['observations'].append(traj_images)
		print("CHECK IF THIS IS (num_traj, traj_len, imsize)")
		import pdb; pdb.set_trace()
	
	save_datasets(recon_dataset, demo_dataset)
	return end


def rollout_trajectory():
	trajectory = get_empty_traj_dict()

	env.reset()
	env_image, traj_images = get_recon_image(env), []
	for j in tqdm(range(num_timesteps)):
		traj_images.append(get_recon_image(env))		
		
		action = spacemouse.get_action()
		observation = env.get_observation()
		next_observation, reward, done, info = env.step(action)

		trajectory['observations'].append(observation)
		trajectory['actions'][j, :] = action
		trajectory['next_observations'].append(next_observation)
		trajectory['rewards'][j] = reward
	return trajectory, env_image, traj_images


recon_dataset = get_empty_recon_dict()
demo_dataset = []

while True:
	print("Trajectory Number:", num_trajectories)
	trajectory, env_image, traj_images = rollout_trajectory()
	end = get_and_process_response()
	num_trajectories += 1
	if end: break