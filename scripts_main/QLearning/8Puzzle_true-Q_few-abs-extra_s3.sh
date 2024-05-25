python train_rl.py \
    --output_path true_Q_output \
    --expt_name few_abs_extra_s3 \
    --env NPuzzle-v0  --env_config "{'size': 3}" \
    --rl_algo QLearning \
    --no_explore \
    --abs_path envinfo_output_main/NPuzzle-v0_N8/few_abs_extra_s3/all_abstractions.json \
    --lr 1.0 \
    --n_episodes 100 \
    --td_moving_avg 1.0 \
    --use_gpu \
    --seed 3
