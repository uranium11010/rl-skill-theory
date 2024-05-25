PYOPENGL_PLATFORM=egl python abstract_trajectories_love.py \
    --env RubiksCube222-v0 \
    --traj_path trajectories/RubiksCube222-v0_traj.pkl \
    --output_path abs_optimal/ \
    --name RubiksCube222_love_s3 \
    --model_config_path abs_scripts/love_configs/RubiksCube222.json \
    --use_abs_pos_kl 1.0 \
    --batch_size 64 \
    --max_iters 20000 \
    --seed 3
