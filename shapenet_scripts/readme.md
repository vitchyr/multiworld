##Shapenet Scripts

#keyboard_control.py
A simple pygame interface that allows you to control the robot with the keyboard:
w,a,s,d are for movements on the horizontal plane
j,k are for vertical movement. j is for moving down and k is for moving up

#randomized_scripted_grasping.py
This scripted collects randomized data and saves the trajectories.

Arguments:
--data_save_directory: Where to store the trajectory and video data. Data is stored in data/data_save_directory/time
--num_trajectories: Number of trajectories to collect
--num_timesteps: Number of timesteps per trajectory
--video_save_frequency: How often to store videos
--gui: Flag indicating whether you want to visualize the randomized grasping.


#scriped_grasping.py
Scripted grasping collects one trajectory using a script. 

Arguments:
--save_video: Option for whether you want to save the video. Video is saved under the directory data/
--gui: Option for whether you want to render the scripted grasping.

#Combine_trajectories.py
Combines trajectories that have been collected in randomized_scripted_grasping.py. 

Arguments:
--data_directory: Specifies the name of the directory inside data where the trajectories are collected
--output_directory: Specifies the name of the directory inside data to output a single file with combined trajectories

#Randomized_grasping_parallel.sh
This script collect data in with multiple trajectories in parallel. It calls the randomized_scripted_grasping.py and combine_trajectories.py
