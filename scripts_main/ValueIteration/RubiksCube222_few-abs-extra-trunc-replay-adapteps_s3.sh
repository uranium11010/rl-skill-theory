python train_rl.py \
    --expt_name few_abs_extra_trunc_replay_adapteps_s3 \
    --env RubiksCube222-v0 \
    --rl_algo ValueIteration \
    --abs_path envinfo_output_main/RubiksCube222-v0/few_abs_extra_s3/all_abstractions.json \
    --truncate_steps 50 \
    --truncate_base_steps 100 \
    --lr 0.1 \
    --n_env_steps 100000000 \
    --early_stop_reward 0.75 \
    --early_stop_model_err 0.1 \
    --model_err_metric wmean_abs_err \
    --test_every 500 \
    --test_every_ratio 1.001 \
    --test_episodes 200 \
    --use_replay_buffer \
    --adaptive_eps_greedy \
    --true_model_path true_V_output/RubiksCube222-v0_ValueIteration_no-expl/few_abs_extra_s3/ \
    --mask_model_errs \
    --save_every 1000 \
    --use_gpu \
    --seed 3