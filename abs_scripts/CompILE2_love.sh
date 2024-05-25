PYOPENGL_PLATFORM=egl python abstract_trajectories_love.py \
    --env CompILE-v0  --env_config "{'visit_length': 2}" \
    --traj_path trajectories/CompILE-v0_s1w10h10n6v2_traj.pkl \
    --output_path abs_optimal/ \
    --name CompILE2_love \
    --model_config_path abs_scripts/love_configs/CompILE2.json \
    --use_abs_pos_kl 1.0 \
    --batch_size 64 \
    --max_iters 20000 \
    --learn_rate 0.0002
