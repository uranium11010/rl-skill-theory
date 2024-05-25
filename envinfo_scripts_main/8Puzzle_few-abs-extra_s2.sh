python compute_envinfo.py \
    --expt_name few_abs_extra_s2 \
    --env NPuzzle-v0  --env_config "{'size': 3}" \
    --n_abs_spaces_per_size 1 5 5 5 5 5 \
    --avg_abs_len 3 \
    --abs_base_action_weights "[0.2, 0.3, 0.3, 0.2]" \
    --extra_abs_path abs_examples/8Puzzle.json \
    --use_gpu \
    --seed 2
