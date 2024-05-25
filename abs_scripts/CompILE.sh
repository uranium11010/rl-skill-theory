python get_trajectories.py \
    --env CompILE-v0 \
    --num_trajectories 1000
mkdir -p abs_optimal
python abstract_trajectories.py \
    --traj_path trajectories/CompILE-v0_traj.pkl \
    --output_path abs_optimal/CompILE.json \
    --action_space_size 5
