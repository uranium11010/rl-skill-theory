import os, shutil
from multiprocessing import pool
import time
from datetime import datetime, timedelta
import argparse
import json
import random

from tqdm import tqdm
import numpy as np
import torch

import gymnasium as gym

from abstractions.abstractions import Axiom, AxiomSeq

from wrappers import RLDiffEnvAdapter, RLDiffEnvAbsWrapper


def generate_abstractions(action_space_size, extra_abs_path, n_abs_spaces_per_size, avg_abs_len, abs_base_action_weights):
    abstractions_list = []
    if extra_abs_path is not None:
        with open(extra_abs_path, 'r') as f:
            extra_abs_lists = json.load(f)
        abstractions_list.extend(extra_abs_lists)
    for num_abs, n_abs_spaces in enumerate(n_abs_spaces_per_size):
        for _ in range(n_abs_spaces):
            random_action_seqs = set()
            while len(random_action_seqs) < num_abs:
                abs_length = np.random.geometric(1 / (avg_abs_len - 1)) + 1
                if isinstance(abs_base_action_weights, dict):
                    abs_base_action_subspaces = list(abs_base_action_weights.keys())
                    (abs_base_action_subspaces_p,
                     abs_base_action_weights_list) = tuple(zip(*abs_base_action_weights.values()))
                    base_action_subspace_idx = np.random.choice(
                            len(abs_base_action_subspaces),
                            p=abs_base_action_subspaces_p)
                    random_action_seq = np.random.choice(
                            abs_base_action_subspaces[base_action_subspace_idx],
                            size=abs_length,
                            p=abs_base_action_weights_list[base_action_subspace_idx])
                else:
                    random_action_seq = np.random.choice(action_space_size, size=abs_length, p=abs_base_action_weights)
                random_action_seqs.add(tuple(action.item() for action in random_action_seq))
            abstractions_list.append(list(random_action_seqs))
    return abstractions_list


def compute_envinfo(args, base_env, run_id, abstraction_int_lists):
    if args.use_gpu:
        torch.set_default_device(run_id % torch.cuda.device_count())
    abstractions = [AxiomSeq([Axiom(action) for action in abstraction_int_list])
                            for abstraction_int_list in abstraction_int_lists]
    env = RLDiffEnvAbsWrapper(base_env, abstractions)
    print(f"RUN {run_id} ENV INFO CALCULATION BEGINS")
    envinfo = {
        "abstractions": abstraction_int_lists,
        "action_space_size": env.action_space.n.item(),
        "num_base_actions": env.num_base_actions,
        "avg_sol_len": env.avg_sol_len,
        "max_sol_len": env.max_sol_len,
        "sum_inv_subopt_gaps_99": env.get_sum_inv_subopt_gaps(0.99),
        "sum_inv_subopt_gaps_95": env.get_sum_inv_subopt_gaps(0.95),
        "sum_inv_subopt_gaps_90": env.get_sum_inv_subopt_gaps(0.90),
        "incompress": env.get_incompress(),
        "expldiff_0": env.get_expldiff(0),
        "expldiff_1e-3": env.get_expldiff(1e-3),
        "expldiff_2e-3": env.get_expldiff(2e-3),
        "expldiff_5e-3": env.get_expldiff(5e-3),
        "expldiff_1e-2": env.get_expldiff(1e-2),
        "expldiff_2e-2": env.get_expldiff(2e-2),
        "expldiff_5e-2": env.get_expldiff(5e-2),
        "expldiff_1e-1": env.get_expldiff(1e-1),
        "expldiff_2e-1": env.get_expldiff(2e-1),
        "expldiff_5e-1": env.get_expldiff(5e-1),
        "expldiff_2e-2_nolog": env.get_expldiff(2e-2, take_log=False),
    }
    print(f"RUN {run_id} ENV INFO:\n{envinfo}")
    with open(os.path.join(args.expt_dir, f"run{run_id}_envinfo.json"), 'w') as f:
        json.dump(envinfo, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute info of base environment and generate macroactions")
    parser.add_argument("--output_path", type=str, default="envinfo_output_main/",
            help="Base directory of output")
    parser.add_argument("--expt_name", type=str, default=datetime.now().strftime("expt_20%y-%m-%d_%Hh%Mm%Ss"),
            help="Name of experiment")
    parser.add_argument("--env", choices=["CliffWalking-v0", "RubiksCube222-v0", "CompILE-v0", "NPuzzle-v0"], required=True,
            help="Name of (base) environment")
    parser.add_argument("--env_config", default="{}",
            help="Environment config as a string containing a Python dictionary")
    parser.add_argument("--seed", type=int, default=0,
            help="Random seed")
    parser.add_argument("--no_parallel", action="store_true",
            help="Run hRL training runs of the same base environment sequentially rather than in parallel")
    parser.add_argument("--num_workers", type=int,
            help="Number of multiprocessing workers")
    parser.add_argument("--use_gpu", action="store_true",
            help="Use GPU(s)")
    parser.add_argument("--ids_to_run", type=str,
            help="String defining Python container object of ids to run")
    parser.add_argument("--overwrite", "-f", action="store_true",
            help="Force overwrite of output directory")
    # Generating macroactions
    parser.add_argument("--n_abs_spaces_per_size", type=int, nargs='+', default=[1],
            help="Number of sets of macroactions to randomly generate for each size of the set; "
                 'e.g., "1 4 6 3" means "1 set of 0 abstractions, 4 sets of 1 abstraction each, '
                 '6 sets of 2 abstractions each, and 3 sets of 3 abstractions each"')
    parser.add_argument("--avg_abs_len", type=float, default=2,
            help="Average macroaction length to randomly generate")
    parser.add_argument("--abs_base_action_weights", type=str,
            help="Frequencies of different actions in randomly generated macroactions; "
                 'as a list, e.g., "[0.25, 0.25, 0.25, 0.25]", '
                 'or as a dict, e.g. {(0, 1): (0.3, [0.5, 0.5]), (2, 3): (0.7, [0.5, 0.5])}')
    parser.add_argument("--extra_abs_path", type=str,
            help="Path to extra macroactions, e.g., optimal ones learnt by LEMMA and those based on them")
    args = parser.parse_args()

    args.env_config = eval(args.env_config)
    args.ids_to_run = list(eval(args.ids_to_run)) if args.ids_to_run is not None else None
    if args.abs_base_action_weights is not None:
        args.abs_base_action_weights = eval(args.abs_base_action_weights)

    torch.set_default_dtype(torch.float64)

    np.random.seed(args.seed)

    base_env = RLDiffEnvAdapter(gym.make(f"{args.env}-rldiff", **args.env_config))
    base_env.transition_table  # this caches the transition table

    args.expt_dir = os.path.join(args.output_path,
                                 args.env
                                 + (f"_{base_env.config_string}" if base_env.config_string is not None else ""),
                                 args.expt_name)
    if os.path.exists(args.expt_dir) and os.listdir(args.expt_dir):
        if args.overwrite:
            print(f"Experiment directory {args.expt_dir} not empty! Removing contents...")
            shutil.rmtree(args.expt_dir, ignore_errors=True)  # remove directory if exists
            time.sleep(5)
        else:
            raise Exception(f"Experiment directory {args.expt_dir} not empty!")
    print(f"Starting experiment in directory {args.expt_dir}")
    os.makedirs(args.expt_dir, exist_ok=True)

    abstractions_list = generate_abstractions(
                            base_env.action_space.n,
                            args.extra_abs_path,
                            args.n_abs_spaces_per_size,
                            args.avg_abs_len,
                            args.abs_base_action_weights
                        )
    print("ABSTRACTIONS:")
    for abstractions in abstractions_list:
        print(abstractions)
    with open(os.path.join(args.expt_dir, f"all_abstractions.json"), 'w') as f:
        json.dump(abstractions_list, f)

    start_time = time.time()
    ids_to_run = range(len(abstractions_list)) if args.ids_to_run is None else args.ids_to_run
    if args.no_parallel:
        for run_id in ids_to_run:
            compute_envinfo(args, base_env, run_id, abstractions_list[run_id])
    else:
        with pool.Pool(args.num_workers, initargs=(tqdm.get_lock(),), initializer=tqdm.set_lock) as p:
            p.starmap(compute_envinfo,
                      [(args, base_env, run_id, abstractions_list[run_id]) for run_id in ids_to_run])
    print("TOTAL TIME:", timedelta(seconds=time.time() - start_time))
