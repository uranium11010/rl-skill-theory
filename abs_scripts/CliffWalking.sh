python get_trajectories.py \
    --env CliffWalking-v0 \
    --num_trajectories 1000
mkdir -p abs_optimal
python abstract_trajectories.py \
    --traj_path trajectories/CliffWalking-v0_traj.pkl \
    --output_path abs_optimal/CliffWalking.json \
    --action_space_size 4
