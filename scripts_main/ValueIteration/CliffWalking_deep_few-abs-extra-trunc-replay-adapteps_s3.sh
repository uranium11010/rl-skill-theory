python train_rl.py \
    --expt_name few_abs_extra_trunc_replay_adapteps_s3 \
    --env CliffWalking-v0 \
    --rl_algo ValueIteration \
    --deep \
    --abs_path envinfo_output_main/CliffWalking-v0/few_abs_extra_s3/all_abstractions.json \
    --truncate_steps 50 \
    --truncate_base_steps 100 \
    --lr 0.0005 \
    --n_env_steps 100000000 \
    --early_stop_reward 0.95 \
    --test_every_ratio 1.001 \
    --use_replay_buffer \
    --adaptive_eps_greedy \
    --save_every 1000 \
    --use_gpu \
    --seed 3
