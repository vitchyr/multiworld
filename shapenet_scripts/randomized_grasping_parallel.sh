for i in {1..2}
do
	python3 randomized_scripted_grasping.py --data_save_directory SawyerGrasp --num_trajectories 10 &
	sleep 1
done
wait

python3 combine_trajectories.py --data_directory SawyerGrasp/trajectories --output_directory SawyerGrasp/consolidated_trajectories
