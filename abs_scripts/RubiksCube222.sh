python get_trajectories.py \
    --env RubiksCube222-v0 \
    --num_trajectories 1000
mkdir -p abs_optimal
python abstract_trajectories.py \
    --traj_path trajectories/RubiksCube222-v0_traj.pkl \
    --output_path abs_optimal/RubiksCube222.json \
    --action_space_size 3 \
    --top 3
