python train_rl.py \
    --output_path true_Q_output \
    --expt_name few_abs_extra \
    --env RubiksCube222-v0 \
    --rl_algo QLearning \
    --no_explore \
    --abs_path envinfo_output_main/RubiksCube222-v0/few_abs_extra/all_abstractions.json \
    --lr 1.0 \
    --n_episodes 100 \
    --td_moving_avg 1.0 \
    --use_gpu
