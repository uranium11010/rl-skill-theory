python train_rl.py \
    --expt_name few_abs_extra_s4 \
    --env NPuzzle-v0  --env_config "{'size': 3}" \
    --rl_algo QLearning \
    --no_explore \
    --abs_path envinfo_output_main/NPuzzle-v0_N8/few_abs_extra_s4/all_abstractions.json \
    --lr 0.1 \
    --early_stop_reward 0.99 \
    --early_stop_model_err 0.001 \
    --model_err_metric wmean_abs_err \
    --test_episodes 200 \
    --true_model_path true_Q_output/NPuzzle-v0_N8_QLearning_no-expl/few_abs_extra_s4/ \
    --use_gpu \
    --seed 4
