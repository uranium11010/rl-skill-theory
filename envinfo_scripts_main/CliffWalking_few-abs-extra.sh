python compute_envinfo.py \
    --expt_name few_abs_extra \
    --env CliffWalking-v0 \
    --n_abs_spaces_per_size 1 5 5 5 5 5 \
    --avg_abs_len 4 \
    --abs_base_action_weights "{(0, 1): (0.4, [0.3, 0.7]),
    				(1, 2): (0.3, [0.7, 0.3]),
				(2, 3): (0.1, [0.7, 0.3]),
				(3, 0): (0.2, [0.3, 0.7])}" \
    --extra_abs_path abs_examples/CliffWalking.json \
    --use_gpu
