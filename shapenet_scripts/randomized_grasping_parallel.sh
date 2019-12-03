for i in {1..2}
do
	python3 randomized_scripted_grasping.py --data-save-directory SawyerGrasp --num-trajectories 10 &
	sleep 1
done
wait

python3 combine_trajectories.py --data-directory SawyerGrasp/trajectories --output-directory SawyerGrasp/consolidated_trajectories
