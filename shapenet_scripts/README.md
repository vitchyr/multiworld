# Shapenet Scripts

## keyboard_control.py
A simple pygame interface that allows you to control the robot with the keyboard:\
w,a,s,d are for movements on the horizontal plane\
j,k are for vertical movement. j is for moving down and k is for moving up

## randomized_scripted_grasping.py
This scripted collects randomized data and saves the trajectories.

### Arguments:
--data_save_directory: Where to store the trajectory and video data. Data is stored in data/data_save_directory/time\
--num_trajectories: Number of trajectories to collect\
--num_timesteps: Number of timesteps per trajectory\
--video_save_frequency: How often to store videos\
--gui: Flag indicating whether you want to visualize the randomized grasping.

### Example:
python randomized_scripted_grasping.py --data_save_directory SawyerGrasp --num_trajectories 100 --num_timesteps 50 --video_save_frequency 1

## scriped_grasping.py
Scripted grasping collects one trajectory using a script. 

### Arguments:
--save_video: Option for whether you want to save the video. Video is saved under the directory data\
--gui: Option for whether you want to render the scripted grasping.

### Example:
python scripted_grasping.py --save_video --gui

## combine_trajectories.py
Combines trajectories that have been collected in randomized_scripted_grasping.py. 

### Arguments:
--data_directory: Specifies the name of the directory inside data where the trajectories are collected

### Example:
python combine_trajectories.py --data_directory SawyerGrasp/trajectories

## randomized_grasping_parallel.py
This script collect data in with multiple trajectories in parallel. It calls the randomized_scripted_grasping.py and combine_trajectories.py

### Example:
python combine_trajectories.py --d sawyer_data --n 1000 -p 10

## sac_rollouts.py
Collects data using a saved SAC model from softlearning. Requires the softlearning package to be installed.

### Example
python shapenet_scripts/sac_rollouts.py /path/to/checkpoint/checkpoint_500/ -d folder_name