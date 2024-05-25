python train_rl.py \
    --output_path true_V_output \
    --expt_name few_abs_extra_s2 \
    --env CliffWalking-v0 \
    --rl_algo ValueIteration \
    --no_explore \
    --abs_path envinfo_output_main/CliffWalking-v0/few_abs_extra_s2/all_abstractions.json \
    --lr 1.0 \
    --n_episodes 100 \
    --td_moving_avg 1.0 \
    --use_gpu \
    --seed 2
