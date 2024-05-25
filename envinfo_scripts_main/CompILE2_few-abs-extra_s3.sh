python compute_envinfo.py \
    --expt_name few_abs_extra_s3 \
    --env CompILE-v0  --env_config "{'visit_length': 2}" \
    --n_abs_spaces_per_size 1 5 5 5 5 5 \
    --avg_abs_len 4 \
    --abs_base_action_weights "{(0, 1, 4): (0.25, [0.4, 0.4, 0.2]),
    				(1, 2, 4): (0.25, [0.4, 0.4, 0.2]),
				(2, 3, 4): (0.25, [0.4, 0.4, 0.2]),
				(3, 0, 4): (0.25, [0.4, 0.4, 0.2])}" \
    --extra_abs_path abs_examples/CompILE2.json \
    --use_gpu \
    --seed 3
