for i in {1..20}
do
	python3 randomized_scripted_grasping.py --video_save_directory videos --data_save_directory trajectories --num_trajectories 10 &
	sleep 1
done
wait

python3 combine_trajectories.py --data_directory trajectories --output_directory consolidated_trajectories
