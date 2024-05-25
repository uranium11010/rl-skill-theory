python train_rl.py \
    --expt_name few_abs_extra_trunc_replay_adapteps_s4 \
    --env CliffWalking-v0 \
    --rl_algo QLearning \
    --abs_path envinfo_output_main/CliffWalking-v0/few_abs_extra_s4/all_abstractions.json \
    --truncate_steps 50 \
    --truncate_base_steps 100 \
    --lr 0.1 \
    --n_env_steps 100000000 \
    --early_stop_reward 0.95 \
    --early_stop_model_err 0.025 \
    --model_err_metric wmean_abs_err \
    --test_every_ratio 1.001 \
    --use_replay_buffer \
    --adaptive_eps_greedy \
    --true_model_path true_Q_output/CliffWalking-v0_QLearning_no-expl/few_abs_extra_s4/ \
    --mask_model_errs \
    --save_every 1000 \
    --use_gpu \
    --seed 4
