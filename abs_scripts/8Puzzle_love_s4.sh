PYOPENGL_PLATFORM=egl python abstract_trajectories_love.py \
    --env NPuzzle-v0  --env_config "{'size': 3}" \
    --traj_path trajectories/NPuzzle-v0_N8_traj.pkl \
    --output_path abs_optimal/ \
    --name 8Puzzle_love_s4 \
    --model_config_path abs_scripts/love_configs/8Puzzle.json \
    --use_abs_pos_kl 1.0 \
    --batch_size 64 \
    --max_iters 20000 \
    --seed 4
