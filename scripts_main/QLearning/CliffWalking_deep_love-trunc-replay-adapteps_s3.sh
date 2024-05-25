python train_rl.py \
    --expt_name love_trunc_replay_adapteps_s3 \
    --env CliffWalking-v0 \
    --rl_algo QLearning \
    --deep --love \
    --love_ckpt_path abs_optimal/CliffWalking_love_s4/model-20000.ckpt \
    --love_model_config_path abs_scripts/love_configs/CliffWalking.json \
    --love_traj_path trajectories/CliffWalking-v0_traj.pkl \
    --truncate_steps 50 \
    --truncate_base_steps 100 \
    --lr 0.0005 \
    --n_env_base_steps 10000000 \
    --early_stop_reward 0.95 \
    --test_every_ratio 1.001 \
    --use_replay_buffer \
    --adaptive_eps_greedy \
    --save_every 1000 \
    --use_gpu \
    --seed 3
