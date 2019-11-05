import roboverse
import numpy as np
import time
import roboverse.utils as utils
import pickle
import os
from PIL import Image, ImageDraw

env = roboverse.make('SawyerGraspOne-v0', render=True)

num_grasps = 0
save_video = True
curr_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(curr_dir)
pklPath = home_dir + '/data'
trajectories = []
image_data = []

for j in range(2000):
    env.reset()
    target_pos = env.get_object_midpoint('duck')
    target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
    images = []
    trajectory = []
    num_timesteps = 100

    for i in range(num_timesteps):
        ee_pos = env.get_end_effector_pos()
        
        grasping_data = []
        grasping_data.append(env.get_observation())

        if i < 50:
            action = target_pos - ee_pos
            action[2] = 0.
            action *= 3.0
            grip=0.
        elif i < 70:
            action = target_pos - ee_pos
            action *= 3.0
            action[2] *= 2.0
            grip=0.
        elif i < 85:
            action = np.zeros((3,))
            grip=1.
        else:
            action = np.zeros((3,))
            action[2] = 1.0
            grip=1.

        action = np.append(action, [grip])
        img = env.render()
        images.append(Image.fromarray(np.uint8(img)))

        next_state, reward, done, info = env.step(action)
        grasping_data.append(next_state)
        grasping_data.append(action)
        grasping_data.append(reward)
        grasping_data.append(done)
        trajectory.append(grasping_data)
        print(grasping_data)

    trajectories.append(trajectory)
    #print(env.render())
    #image_data.append(images)

    object_pos = env.get_object_midpoint('duck')
    if object_pos[2] > -0.1:
        num_grasps += 1

    # TODO write code to save trajectories
    # a list of dictionaries, each dictionary is one trajectory
    # elements of dictionary: np arrays storing state, next_state, action, reward, done
    # can also have images later, so image, next_image, and so on
    
    if j % 10 == 1:
        #with open(pklPath + '/randomized_scripted_duck_gt.p', 'wb+') as fp:
            #pickle.dump(trajectories, fp)

        #with open(pklPath + '/randomized_scripted_duck_images.p', 'wb+') as fp:
            #pickle.dump(image_data, fp)
        print("HI")
        if save_video:
            images[0].save('dump/grasp_duck_randomized/{}.gif'.format(j), format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)
            #utils.save_video('dump/grasp_duck_randomized/{}.avi'.format(j), images)

    print('Num attempts: {}'.format(j))
    print('Num grasps: {}'.format(num_grasps))

