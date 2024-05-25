python get_trajectories.py \
    --env CompILE-v0  --env_config "{'visit_length': 2}" \
    --num_trajectories 1000
mkdir -p abs_optimal
python abstract_trajectories.py \
    --traj_path trajectories/CompILE-v0_s1w10h10n6v2_traj.pkl \
    --output_path abs_optimal/CompILE2.json \
    --action_space_size 5
