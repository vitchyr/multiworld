for i in {1..20}
do
	python3 randomized_scripted_grasping.py --video_save_directory videos --data_save_directory trajectories --num_trajectories 100 &
	sleep 1
done
wait
