import roboverse
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_save_directory", type=str)
parser.add_argument("--num_trajectories", type=int, default=2000)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")
parser.add_argument("--gui", dest="gui", action="store_true", default=False)

args = parser.parse_args()
timestamp = roboverse.utils.timestamp()
data_save_path = os.path.join(__file__, "../..", 'data', 'SawyerGrasp',
                              args.data_save_directory + "_" + timestamp)
data_save_path = os.path.abspath(data_save_path)
video_save_path = os.path.join(data_save_path, "videos")

env = roboverse.make('SawyerGraspOne-v0', gui=args.gui)
object_name = 'lego'
num_grasps = 0
image_data = []

obs_dim = env.observation_space.shape
assert(len(obs_dim) == 1)
obs_dim = obs_dim[0]
act_dim = env.action_space.shape[0]

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)
if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
    os.makedirs(video_save_path)


pool = roboverse.utils.DemoPool()

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

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)

    object_pos = env.get_object_midpoint(object_name)
    if object_pos[2] > -0.1:
        num_grasps += 1
        print('Num grasps: {}'.format(num_grasps))

    if args.video_save_frequency > 0 and j % args.video_save_frequency == 0:
        images[0].save('{}/{}.gif'.format(video_save_path, j),
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)

params = env.get_params()
pool.save(params, data_save_path,
          '{}_pool_{}.pkl'.format(timestamp, pool.size))
