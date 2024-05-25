python train_rl.py \
    --expt_name few_abs_extra_trunc_replay_adapteps_s4 \
    --env NPuzzle-v0  --env_config "{'size': 3}" \
    --rl_algo ValueIteration \
    --deep \
    --abs_path envinfo_output_main/NPuzzle-v0_N8/few_abs_extra_s4/all_abstractions.json \
    --truncate_steps 50 \
    --truncate_base_steps 100 \
    --lr 0.0005 \
    --n_env_steps 100000000 \
    --early_stop_reward 0.95 \
    --test_every 500 \
    --test_every_ratio 1.001 \
    --test_episodes 200 \
    --use_replay_buffer \
    --adaptive_eps_greedy \
    --save_every 1000 \
    --use_gpu \
    --seed 4
