python get_trajectories.py \
    --env NPuzzle-v0  --env_config "{'size': 3}" \
    --num_trajectories 1000
mkdir -p abs_optimal
python abstract_trajectories.py \
    --traj_path trajectories/NPuzzle-v0_N8_traj.pkl \
    --output_path abs_optimal/8Puzzle.json \
    --action_space_size 4 \
    --top 2
