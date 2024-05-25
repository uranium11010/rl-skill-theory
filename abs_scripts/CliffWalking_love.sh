PYOPENGL_PLATFORM=egl python abstract_trajectories_love.py \
    --env CliffWalking-v0 \
    --traj_path trajectories/CliffWalking-v0_traj.pkl \
    --output_path abs_optimal/ \
    --name CliffWalking_love \
    --model_config_path abs_scripts/love_configs/CliffWalking.json \
    --use_abs_pos_kl 1.0 \
    --batch_size 64 \
    --max_iters 20000 \
    --learn_rate 0.0001
