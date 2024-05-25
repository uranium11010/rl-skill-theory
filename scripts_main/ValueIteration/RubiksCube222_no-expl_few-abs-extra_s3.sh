python train_rl.py \
    --expt_name few_abs_extra_s3 \
    --env RubiksCube222-v0 \
    --rl_algo ValueIteration \
    --no_explore \
    --abs_path envinfo_output_main/RubiksCube222-v0/few_abs_extra_s3/all_abstractions.json \
    --lr 0.1 \
    --early_stop_reward 0.99 \
    --early_stop_model_err 0.001 \
    --model_err_metric wmean_abs_err \
    --test_episodes 200 \
    --true_model_path true_V_output/RubiksCube222-v0_ValueIteration_no-expl/few_abs_extra_s3/ \
    --use_gpu \
    --seed 3