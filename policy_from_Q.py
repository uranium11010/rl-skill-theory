import os
import argparse
from tqdm import tqdm

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--true_q_path", required=True)
    parser.add_argument("--true_pi_path", required=True)
    args = parser.parse_args()

    os.makedirs(args.true_pi_path, exist_ok=True)
    for run_id in tqdm(range(args.num_runs)):
        true_q_values = np.load(os.path.join(args.true_q_path, f"run{run_id}_Q_values.npy"))
        assert len(true_q_values.shape) == 2
        true_pi_actions = true_q_values == np.max(true_q_values, axis=1, keepdims=True)
        np.save(os.path.join(args.true_pi_path, f"run{run_id}_pi_actions.npy"), true_pi_actions)
