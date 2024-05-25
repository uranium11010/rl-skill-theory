python train_rl.py \
    --expt_name few_abs_extra_trunc_s4 \
    --env RubiksCube222-v0 \
    --rl_algo REINFORCE \
    --abs_path envinfo_output_main/RubiksCube222-v0/few_abs_extra_s4/all_abstractions.json \
    --truncate_steps 50 \
    --truncate_base_steps 100 \
    --lr 0.1 \
    --n_env_steps 100000000 \
    --early_stop_reward 0.75 \
    --test_every 500 \
    --test_every_ratio 1.001 \
    --test_episodes 200 \
    --test_no_greedy \
    --true_model_path true_pi_output/RubiksCube222-v0_QLearning_no-expl/few_abs_extra_s4/ \
    --mask_model_errs \
    --model_err_metric wmean_kl \
    --save_every 1000 \
    --use_gpu \
    --seed 4
