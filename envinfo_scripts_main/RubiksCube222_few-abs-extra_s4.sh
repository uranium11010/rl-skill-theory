python compute_envinfo.py \
    --expt_name few_abs_extra_s4 \
    --env RubiksCube222-v0 \
    --n_abs_spaces_per_size 1 5 5 5 5 5 \
    --avg_abs_len 3 \
    --extra_abs_path abs_examples/RubiksCube222.json \
    --use_gpu \
    --seed 4
